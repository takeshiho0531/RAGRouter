import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import argparse
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from utils import json2dict


def load_scores_and_data(jsonl_path):
    print(f"Loading retrieval results from {jsonl_path}...")
    scores_list = []
    query_ids = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            q_id = data.get("id", data.get("question_id"))
            scores = [float(ctx.get("retrieval score", 0)) for ctx in data["ctxs"]]
            while len(scores) < 5:
                scores.append(0.0)
            scores_list.append(scores[:5])
            query_ids.append(str(q_id))
    return query_ids, np.array(scores_list)


class FeatureExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)

    def get_features(
        self,
        queries,
        docs,
        scores_np,
        ce_scores_top5,
        query_emb_pca=None,
        doc_emb_pca=None,
    ):
        ce_top1 = ce_scores_top5[:, 0].reshape(-1, 1)
        ce_max = np.max(ce_scores_top5, axis=1).reshape(-1, 1)

        agreement_score = (scores_np[:, 0].reshape(-1, 1)) * ce_top1

        q_char_len = np.array([len(q) for q in queries]).reshape(-1, 1)
        q_word_count = np.array([len(q.split()) for q in queries]).reshape(-1, 1)

        features_list = [ce_top1, ce_max, agreement_score, q_char_len, q_word_count]

        if query_emb_pca is not None:
            features_list.append(query_emb_pca)
        if doc_emb_pca is not None:
            features_list.append(doc_emb_pca)

        return np.hstack(features_list)


def prepare_data(args, extractor):
    jsonl_path = f"data/{args.data_type}/triviaqa::olmes_q_retrieved_results::_IVFPQ.65536.64.256::k5.jsonl"
    qid_list, all_scores = load_scores_and_data(jsonl_path)
    score_map = dict(zip(qid_list, all_scores))

    query_map = json2dict(f"data/{args.data_type}/query_map.json")
    doc_map = json2dict(f"data/{args.data_type}/doc_map.json")

    dfs = {}
    all_queries_text = []
    all_docs_text = []

    for split in ["train", "test"]:
        df = pd.read_csv(f"data/{args.data_type}/{split}_{args.task_type}.csv")
        df["query_id_str"] = df["query_id"].astype(str)
        dfs[split] = df
        all_queries_text.extend([query_map.get(str(qid), "") for qid in df["query_id"]])
        all_docs_text.extend([doc_map.get(str(did), "") for did in df["doc_id"]])

    print("Encoding embeddings for PCA...")
    all_query_embs = extractor.model.encode(all_queries_text, show_progress_bar=True)
    all_doc_embs = extractor.model.encode(all_docs_text, show_progress_bar=True)

    n_q_pca = 64
    n_d_pca = 16
    print(f"Computing PCA (Q:{n_q_pca}, D:{n_d_pca})...")
    pca_query = PCA(n_components=n_q_pca)
    pca_doc = PCA(n_components=n_d_pca)
    all_queries_pca = pca_query.fit_transform(all_query_embs)
    all_docs_pca = pca_doc.fit_transform(all_doc_embs)

    start_idx = 0
    datasets = {}
    for split in ["train", "test"]:
        df = dfs[split]
        num_samples = len(df)
        split_q_pca = all_queries_pca[start_idx : start_idx + num_samples]
        split_d_pca = all_docs_pca[start_idx : start_idx + num_samples]
        start_idx += num_samples

        ce_path = f"data/{args.data_type}/ce_score_top5_{'train_stacking' if split == 'train' else 'all'}.csv"
        ce_df = pd.read_csv(ce_path)
        ce_df["query_id"] = ce_df["query_id"].astype(str)

        merged_df = pd.merge(
            df, ce_df, left_on="query_id_str", right_on="query_id", how="left"
        )
        ce_scores_top5 = (
            merged_df[[f"ce_score_{i}" for i in range(1, 6)]].fillna(0.0).values
        )
        queries = [query_map.get(str(qid), "") for qid in merged_df["query_id_str"]]
        docs = [doc_map.get(str(did), "") for did in merged_df["doc_id"]]
        batch_scores = np.array(
            [score_map.get(qid, [0.0] * 5) for qid in merged_df["query_id_str"]]
        )

        l_rag = merged_df["0"].values.astype(float)
        norag_df = pd.read_csv(f"data/{args.data_type}/{split}.csv")
        l_norag = norag_df["1"].values.astype(float)

        X = extractor.get_features(
            queries,
            docs,
            batch_scores,
            ce_scores_top5,
            query_emb_pca=split_q_pca,
            doc_emb_pca=split_d_pca,
        )
        datasets[split] = {"X": X, "l_rag": l_rag, "l_norag": l_norag}
    return datasets


def train_lightgbm(datasets, args):
    print("\n=== Training Targeted Router (High-Dim PCA) ===")
    X_train = datasets["train"]["X"]
    y_train = (
        (datasets["train"]["l_rag"] == 1.0) & (datasets["train"]["l_norag"] == 0.0)
    ).astype(int)

    weights = np.where(y_train == 1, 8.0, 1.0)
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.01,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        "lambda_l1": 2.0,
        "lambda_l2": 2.0,
        "verbose": -1,
    }

    gbm = lgb.train(params, train_data, num_boost_round=1500)

    n_q_pca = 64
    n_d_pca = 16
    feature_names = ["ce_top1", "ce_max", "agreement", "q_char_len", "q_word_count"]
    feature_names += [f"q_pca_{i+1}" for i in range(n_q_pca)]
    feature_names += [f"d_pca_{i+1}" for i in range(n_d_pca)]

    print("\n=== Feature Importance (Top 10) ===")
    importance = gbm.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)
    print(feat_imp.head(10))

    y_probs_train = gbm.predict(X_train)
    best_thr = 0.5
    max_acc = 0
    for thr in np.arange(0.1, 0.9, 0.01):
        preds = (y_probs_train > thr).astype(int)
        rec = recall_score(y_train, preds)
        acc = accuracy_score(y_train, preds)
        if rec >= 0.65:
            if acc > max_acc:
                max_acc = acc
                best_thr = thr

    print(f"\nBest Threshold (Recall-Focused): {best_thr:.3f}")
    evaluate_router(gbm, datasets["test"], threshold=best_thr)


def evaluate_router(model, test_data, threshold=0.5):
    X_test = test_data["X"]
    l_rag, l_norag = test_data["l_rag"], test_data["l_norag"]
    probs = model.predict(X_test)
    preds_is_rag = probs > threshold

    # evaluate router's classification performance
    y_true_router = ((l_rag == 1.0) & (l_norag == 0.0)).astype(int)
    y_pred_router = preds_is_rag.astype(int)

    print("\n=== Router Classification Performance (Is it a 14B miss?) ===")
    # Target_Miss: 14B_Miss_RAG_Save,
    # Other: Other
    print(
        classification_report(
            y_true_router, y_pred_router, target_names=["Other", "14B_Miss_RAG_Save"]
        )
    )

    # evaluate system-level performance (accuracy and recall for the entire evaluation dataset)
    miss_mask = y_true_router == 1
    saved = sum(preds_is_rag & miss_mask)
    total_miss = sum(miss_mask)

    print("=== System Level Evaluation ===")
    print(f"Recall (Saved 14B misses): {saved/total_miss:.4f} ({saved}/{total_miss})")

    correct = 0
    for i in range(len(l_rag)):
        if (l_rag[i] if preds_is_rag[i] else l_norag[i]) == 1.0:
            correct += 1
    print(f"Total System Accuracy: {correct / len(l_rag):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="triviaqa_full")
    parser.add_argument("--task_type", type=str, default="local")
    args = parser.parse_args()
    extractor = FeatureExtractor()
    datasets = prepare_data(args, extractor)
    train_lightgbm(datasets, args)
