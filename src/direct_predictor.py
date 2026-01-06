import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import argparse
import os
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    matthews_corrcoef,
    f1_score,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

PCA_Q_DIM = 32
PCA_D_DIM = 16


def json2dict(path):
    with open(path, "r") as f:
        return json.load(f)


def load_scores_and_data(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"[WARN] retrieval jsonl not found: {jsonl_path}")
        return [], np.zeros((0, 5), dtype=np.float32)

    scores_list = []
    query_ids = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            q_id = data.get("id", data.get("question_id"))
            if q_id is None:
                continue
            ctxs = data.get("ctxs", [])
            scores = [float(ctx.get("retrieval score", 0)) for ctx in ctxs]
            while len(scores) < 5:
                scores.append(0.0)
            scores_list.append(scores[:5])
            query_ids.append(str(q_id))

    return query_ids, np.array(scores_list, dtype=np.float32)


class FeatureExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)

    def get_features(self, queries, scores_np, ce_scores_top5, q_pca=None, d_pca=None):
        ce_top1 = ce_scores_top5[:, 0].reshape(-1, 1)
        ce_max = np.max(ce_scores_top5, axis=1).reshape(-1, 1)
        ce_diff = (ce_scores_top5[:, 0] - ce_scores_top5[:, 1]).reshape(-1, 1)

        ret_diff = (scores_np[:, 0] - scores_np[:, 1]).reshape(-1, 1)

        q_len = np.array([len(q) for q in queries], dtype=np.float32).reshape(-1, 1)
        q_words = np.array([len(q.split()) for q in queries], dtype=np.float32).reshape(
            -1, 1
        )

        agreement = (scores_np[:, 0].reshape(-1, 1)) * ce_top1

        features = [ce_top1, ce_max, ce_diff, ret_diff, q_len, q_words, agreement]
        if q_pca is not None:
            features.append(q_pca)
        if d_pca is not None:
            features.append(d_pca)

        return np.hstack(features).astype(np.float32)


def prepare_datasets(args, extractor):
    base = f"data/{args.data_type}"

    jsonl_path = (
        f"{base}/triviaqa::olmes_q_retrieved_results::_IVFPQ.65536.64.256::k5.jsonl"
    )
    qid_list, all_scores = load_scores_and_data(jsonl_path)
    score_map = dict(zip(qid_list, all_scores))

    query_map = json2dict(f"{base}/query_map.json")
    doc_map = json2dict(f"{base}/doc_map.json")

    pca_q = PCA(n_components=PCA_Q_DIM)
    pca_d = PCA(n_components=PCA_D_DIM)

    raw_data = {}
    for split in ["train", "test"]:
        df = pd.read_csv(f"{base}/{split}_local.csv")
        df["query_id_str"] = df["query_id"].astype(str)

        q_text = [query_map.get(str(qid), "") for qid in df["query_id"]]
        d_text = [doc_map.get(str(did), "") for did in df["doc_id"]]

        q_emb = extractor.model.encode(q_text, show_progress_bar=True)
        d_emb = extractor.model.encode(d_text, show_progress_bar=True)

        raw_data[split] = {
            "df": df,
            "q_text": q_text,
            "q_emb": q_emb,
            "d_emb": d_emb,
        }

    print("Fitting PCA on Train data only...")
    pca_q.fit(raw_data["train"]["q_emb"])
    pca_d.fit(raw_data["train"]["d_emb"])

    datasets = {}
    for split in ["train", "test"]:
        split_data = raw_data[split]
        df = split_data["df"]

        q_pca = pca_q.transform(split_data["q_emb"]).astype(np.float32)
        d_pca = pca_d.transform(split_data["d_emb"]).astype(np.float32)

        ce_path = f"{base}/ce_score_top5_{'train_stacking' if split == 'train' else 'all'}.csv"
        ce_df = pd.read_csv(ce_path)
        ce_df["query_id"] = ce_df["query_id"].astype(str)

        merged = pd.merge(
            df, ce_df, left_on="query_id_str", right_on="query_id", how="left"
        )

        ce_scores = (
            merged[[f"ce_score_{i}" for i in range(1, 6)]].fillna(0.0).values
        ).astype(np.float32)

        batch_scores = np.array(
            [score_map.get(qid, [0.0] * 5) for qid in merged["query_id_str"]],
            dtype=np.float32,
        )

        X = extractor.get_features(
            split_data["q_text"], batch_scores, ce_scores, q_pca=q_pca, d_pca=d_pca
        )

        y_rag = merged["0"].values.astype(np.float32)
        y_14b = pd.read_csv(f"{base}/{split}.csv")["1"].values.astype(np.float32)

        if len(y_14b) != len(y_rag):
            raise ValueError(
                f"[{split}] length mismatch: y_14b={len(y_14b)} vs y_rag={len(y_rag)} "
                f"(expected they align by row order as in your pipeline)"
            )

        y_needrag = ((y_14b == 0.0) & (y_rag == 1.0)).astype(np.int32)

        datasets[split] = {
            "X": X,
            "y_14b": y_14b,
            "y_rag": y_rag,
            "y_needrag": y_needrag,
        }

    f_names = [
        "ce_top1",
        "ce_max",
        "ce_diff",
        "ret_diff",
        "q_len",
        "q_words",
        "agreement",
    ]
    f_names += [f"q_pca_{i}" for i in range(PCA_Q_DIM)]
    f_names += [f"d_pca_{i}" for i in range(PCA_D_DIM)]
    return datasets, f_names


class NeedRAGRouter:
    def __init__(self):
        self.model = None
        self.best_thr = 0.5
        self.feature_names = None

        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.02,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "n_jobs": -1,
        }

    def train(self, X_train, y_needrag, feature_names, seed=42):
        self.feature_names = feature_names

        X_t, X_v, y_t, y_v = train_test_split(
            X_train, y_needrag, test_size=0.15, random_state=seed, stratify=y_needrag
        )

        pos_weight = 15.0  # Imbalance handling: weight positives higher
        w_t = np.where(y_t == 1, pos_weight, 1.0).astype(np.float32)
        w_v = np.where(y_v == 1, pos_weight, 1.0).astype(np.float32)

        ds_train = lgb.Dataset(X_t, label=y_t, weight=w_t)
        ds_valid = lgb.Dataset(X_v, label=y_v, weight=w_v, reference=ds_train)

        print("Training NeedRAG Predictor (14B fails & 0.6B+RAG succeeds)...")
        self.model = lgb.train(
            self.params,
            ds_train,
            num_boost_round=5000,
            valid_sets=[ds_valid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )

        # self.best_thr = self._search_threshold(
        #     X_v, y_v, objective="recall_at_precision", min_precision=0.60
        # )
        self.best_thr = self._search_threshold(X_v, y_v, objective="f1")
        print(f"Best threshold: {self.best_thr:.3f}")

    def _search_threshold(
        self,
        X_val,
        y_val,
        objective="recall_at_precision",
        min_precision=0.60,
    ):
        p = self.model.predict(X_val, num_iteration=self.model.best_iteration)

        best_thr = 0.5
        best_val = -1.0

        for thr in np.arange(0.01, 0.99, 0.001):
            pred = (p > thr).astype(np.int32)
            prec = precision_score(y_val, pred, zero_division=0)
            rec = recall_score(y_val, pred, zero_division=0)
            f1 = f1_score(y_val, pred, zero_division=0)

            if objective == "f1":
                val = f1
                if val > best_val:
                    best_val = val
                    best_thr = float(thr)
            elif objective == "recall_at_precision":
                if prec >= min_precision:
                    val = rec
                    if val > best_val:
                        best_val = val
                        best_thr = float(thr)
            else:
                raise ValueError(f"unknown objective: {objective}")

        return best_thr

    def predict_route(self, X):
        p = self.model.predict(X, num_iteration=self.model.best_iteration)
        route = p > self.best_thr
        return route.astype(bool), p

    def show_importance(self, topk=10):
        if self.model is None:
            return
        print(f"--- Feature Importance (NeedRAG) Top {topk} ---")
        imp = pd.DataFrame(
            {
                "f": self.feature_names,
                "v": self.model.feature_importance(importance_type="gain"),
            }
        )
        print(imp.sort_values("v", ascending=False).head(topk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="triviaqa_full")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--min_precision",
        type=float,
        default=0.60,
        help="threshold search: maximize recall s.t. precision >= this",
    )
    args = parser.parse_args()

    extractor = FeatureExtractor(device=args.device)
    data, f_names = prepare_datasets(args, extractor)

    router = NeedRAGRouter()
    router.train(
        data["train"]["X"],
        data["train"]["y_needrag"],
        f_names,
    )

    route, p_needrag = router.predict_route(data["test"]["X"])
    y14 = data["test"]["y_14b"]
    yrag = data["test"]["y_rag"]
    y_needrag = data["test"]["y_needrag"]

    final_res = np.where(route, yrag, y14)
    system_acc = float(np.mean(final_res))
    call_rate = float(np.mean(route))

    router_preds = route.astype(int)
    r_acc = accuracy_score(y_needrag, router_preds)
    r_pre = precision_score(y_needrag, router_preds, zero_division=0)
    r_rec = recall_score(y_needrag, router_preds, zero_division=0)
    r_mcc = matthews_corrcoef(y_needrag, router_preds)

    print(f"=== System Performance Evaluation ===")
    print(f"Total System Accuracy: {system_acc:.4f} (Baseline 14B: {np.mean(y14):.4f})")
    print(f"RAG Call Rate: {call_rate:.2%}")

    print(f"=== Router Decision Quality (NeedRAG detection) ===")
    print(f"Router Selection Accuracy: {r_acc:.4f}")
    print(f"├─ Precision (effective RAG): {r_pre:.4f}")
    print(f"├─ Recall    (rescue rate):   {r_rec:.4f}")
    print(f"└─ MCC       (overall):       {r_mcc:.4f}")

    cm = confusion_matrix(y_needrag, router_preds)
    print(f"Confusion Matrix (NeedRAG target):")
    print(f"                Pred: noRAG | Pred: RAG")
    print(f"Actual: noNeed: {cm[0,0]:>7} | {cm[0,1]:>9} (Overkill/Waste)")
    print(f"Actual: NeedRAG:{cm[1,0]:>7} | {cm[1,1]:>9} (Saved!)")

    router.show_importance(topk=10)


if __name__ == "__main__":
    main()
