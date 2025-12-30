import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import argparse
from utils import json2dict
import json
from sklearn.decomposition import PCA


def load_scores_and_data(jsonl_path):
    scores_list = []
    query_ids = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            scores = [float(ctx.get("retrieval score", 0)) for ctx in data["ctxs"]]
            while len(scores) < 5:
                scores.append(0.0)
            scores_list.append(scores[:5])
            query_ids.append(data["question_id"])
    return query_ids, np.array(scores_list)


class FeatureExtractor:
    def __init__(
        self,
        train_queries=None,
        train_docs=None,
        model_name="all-MiniLM-L6-v2",
        device="cuda",
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.word_idf = {}

    #     if train_queries is not None and train_docs is not None:
    #         self._compute_idf(train_queries + train_docs)

    # def _compute_idf(self, texts):
    #     print("Computing IDF for lexical features...")
    #     doc_count = len(texts)
    #     word_counter = Counter()
    #     for text in texts:
    #         words = set(text.lower().split())
    #         word_counter.update(words)
    #     for word, freq in word_counter.items():
    #         self.word_idf[word] = math.log(doc_count / (freq + 1))

    # def _get_lexical_features(self, queries, docs):
    #     features = []
    #     for q, d in zip(queries, docs):
    #         q_tokens = set(q.lower().split())
    #         d_tokens = set(d.lower().split())
    #         if len(q_tokens) == 0:
    #             features.append([0.0, 0.0, 0.0, 0.0])
    #             continue
    #         intersection = q_tokens.intersection(d_tokens)
    #         union = q_tokens.union(d_tokens)
    #         jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    #         len_ratio = len(d) / len(q) if len(q) > 0 else 0
    #         idf_overlap_score = sum(self.word_idf.get(w, 0) for w in intersection)
    #         q_total_idf = sum(self.word_idf.get(w, 0) for w in q_tokens)
    #         weighted_overlap_ratio = (idf_overlap_score / q_total_idf if q_total_idf > 0 else 0)
    #         features.append([jaccard, len_ratio, idf_overlap_score, weighted_overlap_ratio])
    #     return np.array(features)

    def get_features(
        self,
        queries,
        docs,
        scores_np,
        ce_scores_top5,
        query_emb_pca=None,
        doc_emb_pca=None,
    ):
        # cross-encoder scores
        ce_top1 = ce_scores_top5[:, 0].reshape(-1, 1)
        ce_max = np.max(ce_scores_top5, axis=1).reshape(-1, 1)

        agreement_score = (scores_np[:, 0].reshape(-1, 1)) * ce_top1

        q_len = np.array([len(q) for q in queries]).reshape(-1, 1)
        q_pca_limited = query_emb_pca[:, :8] if query_emb_pca is not None else None

        features_list = [
            ce_top1,  # 0
            ce_max,  # 1
            agreement_score,  # 2
            q_len,  # 3
        ]

        if q_pca_limited is not None:
            features_list.append(q_pca_limited)  # 4-11
        if doc_emb_pca is not None:
            features_list.append(doc_emb_pca)  # 12-27

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
        df["original_query_id"] = df["query_id"]
        df["query_id"] = (
            df["query_id"].astype(str).str.replace("triviaqa::olmes:", "", regex=False)
        )
        dfs[split] = df

        all_queries_text.extend(
            [query_map.get(str(qid), "") for qid in df["original_query_id"]]
        )
        all_docs_text.extend([doc_map.get(str(did), "") for did in df["doc_id"]])

    print("Encoding all queries and docs for PCA...")
    all_query_embs = extractor.model.encode(all_queries_text, show_progress_bar=True)
    all_doc_embs = extractor.model.encode(all_docs_text, show_progress_bar=True)

    print("Computing PCA components...")
    pca_query = PCA(n_components=8)
    pca_doc = PCA(n_components=16)
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
        merged_df = pd.merge(df, ce_df, on="query_id", how="left")
        ce_scores_top5 = (
            merged_df[[f"ce_score_{i}" for i in range(1, 6)]].fillna(0.0).values
        )
        queries = [
            query_map.get(str(qid), "") for qid in merged_df["original_query_id"]
        ]
        docs = [doc_map.get(str(did), "") for did in merged_df["doc_id"]]
        batch_scores = np.array(
            [
                score_map.get(str(qid).replace("triviaqa::olmes:", ""), [0] * 5)
                for qid in merged_df["original_query_id"]
            ]
        )  # "retrieval scores"
        l_rag = merged_df["0"].values.astype(float)

        print(f"Extracting features for {split} with Q&D PCA...")
        X = extractor.get_features(
            queries,
            docs,
            batch_scores,
            ce_scores_top5,
            query_emb_pca=split_q_pca,
            doc_emb_pca=split_d_pca,
        )
        datasets[split] = {"X": X, "l_rag": l_rag}

    return datasets


def train_lightgbm(datasets, query_map, doc_map, test_rag_df):
    print("=== Training Stacking Model ===")
    X_train, y_train = datasets["train"]["X"], (
        datasets["train"]["l_rag"] == 1.0
    ).astype(int)

    X_train_noise = X_train.copy()
    X_train_noise[:, 0:2] += np.random.normal(0, 0.01, X_train_noise[:, 0:2].shape)

    train_data = lgb.Dataset(X_train_noise, label=y_train)

    params = {
        "objective": "binary",
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.01,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "lambda_l1": 10.0,
        "lambda_l2": 10.0,
        "verbose": -1,
    }

    gbm = lgb.train(params, train_data, num_boost_round=5000)

    # optimize threshold on train set
    y_probs_train = gbm.predict(X_train)
    best_thr, best_acc = 0.5, 0.0
    for thr in np.arange(0.1, 0.9, 0.005):
        acc = accuracy_score(y_train, (y_probs_train > thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    print(f"Best Threshold: {best_thr:.3f} (Train Acc: {best_acc:.4f})")
    print("=== Test Evaluation ===")
    evaluate_router(gbm, datasets["test"], threshold=best_thr)

    # Detailed analysis
    # importance features analysis
    feature_names = [
        "ce_top1",
        "ce_max",
        "agreement_score",
        "q_len",
    ]
    feature_names += [f"q_pca_{i+1}" for i in range(8)]
    feature_names += [f"d_pca_{i+1}" for i in range(16)]
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": gbm.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    print("--- Feature Importance (Top 10) ---")
    print(importances.head(10))

    # hard false positives analysis
    X_test = datasets["test"]["X"]
    y_probs_test = gbm.predict(X_test)
    y_true_test = (datasets["test"]["l_rag"] == 1.0).astype(int)
    hard_fp = np.where((y_probs_test > 0.7) & (y_true_test == 0))[0]
    print(f"--- Hard False Positives Analysis (Total: {len(hard_fp)}) ---")
    for idx in hard_fp[:5]:
        row = test_rag_df.iloc[idx]
        q_id = str(row["query_id"])
        d_id = str(row["doc_id"])
        q_text = query_map.get(
            q_id, query_map.get(f"triviaqa::olmes:{q_id}", "Unknown")
        )
        d_text = doc_map.get(d_id, "Unknown")

        print(f"[Index: {idx}] how much router is confident: {y_probs_test[idx]:.4f}")
        print(f"Query: {q_text}")
        print(f"Doc (Top-1): {d_text[:100]}...")
        print(f"CE_top1 score: {X_test[idx, 0]:.4f}")
        print("-" * 60)

    return gbm


def evaluate_router(model, test_data, threshold=0.5):
    X_test, y_true = test_data["X"], (test_data["l_rag"] == 1.0).astype(int)
    probs = model.predict(X_test)
    preds = (probs > threshold).astype(int)
    print(
        classification_report(y_true, preds, target_names=["RAG_Wrong", "RAG_Correct"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="triviaqa_20000")
    parser.add_argument("--task_type", type=str, default="local")
    args = parser.parse_args()

    query_map = json2dict(f"data/{args.data_type}/query_map.json")
    doc_map = json2dict(f"data/{args.data_type}/doc_map.json")

    test_rag_df = pd.read_csv(f"data/{args.data_type}/test_{args.task_type}.csv")

    extractor = FeatureExtractor(
        train_queries=list(query_map.values()), train_docs=list(doc_map.values())
    )

    datasets = prepare_data(args, extractor)
    train_lightgbm(datasets, query_map, doc_map, test_rag_df)
