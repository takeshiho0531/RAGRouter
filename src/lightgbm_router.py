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
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


def json2dict(path):
    with open(path, "r") as f:
        return json.load(f)


def load_scores_and_data(jsonl_path):
    if not os.path.exists(jsonl_path):
        return [], np.zeros((0, 5))
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

    def get_features(self, queries, scores_np, ce_scores_top5, query_emb_pca=None):
        ce_top1 = ce_scores_top5[:, 0].reshape(-1, 1)
        ce_max = np.max(ce_scores_top5, axis=1).reshape(-1, 1)

        # retrieveal confidence features: whether the top retrieved context is clearly better than the second best
        ce_diff = (ce_scores_top5[:, 0] - ce_scores_top5[:, 1]).reshape(-1, 1)
        ret_diff = (scores_np[:, 0] - scores_np[:, 1]).reshape(-1, 1)

        # how difficult is the query?
        q_len = np.array([len(q) for q in queries]).reshape(-1, 1)
        q_words = np.array([len(q.split()) for q in queries]).reshape(-1, 1)

        features_list = [ce_top1, ce_max, ce_diff, ret_diff, q_len, q_words]

        if query_emb_pca is not None:
            features_list.append(query_emb_pca)

        return np.hstack(features_list)


class CascadeRouter:
    def __init__(
        self, trust_14b_threshold=0.85
    ):  # strict threshold so that 14B is trusted often
        self.model_14b = None
        self.model_06b_rag = None
        self.trust_threshold = trust_14b_threshold
        self.best_delta = 0.0
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.005,
            "feature_fraction": 0.8,
            "verbose": -1,
        }

    def train(self, X_train, y_14b, y_rag):
        X_t, X_v, y14_t, y14_v, yrag_t, yrag_v = train_test_split(
            X_train, y_14b, y_rag, test_size=0.15, random_state=42
        )

        print("Training 14B Success Predictor...")
        self.model_14b = lgb.train(self.params, lgb.Dataset(X_t, label=y14_t), 2000)

        print("Training 0.6B+RAG Success Predictor...")
        target_mask = (y14_t == 0) & (yrag_t == 1)
        sample_weights = np.where(target_mask, 15.0, 1.0)  # ターゲットへの執着を強化
        self.model_06b_rag = lgb.train(
            self.params, lgb.Dataset(X_t, label=yrag_t, weight=sample_weights), 2000
        )

        self._optimize_router_strict(X_v, y14_v, yrag_v)

    def _optimize_router_strict(self, X_val, y14_val, yrag_val):
        """
        Reduce overkill (a case using RAG though it's useless) while maximizing the Router's selection accuracy (14B vs RAG)
        """
        print("Optimizing for Strict Selection Accuracy...")
        X_val_14b = X_val.copy()
        X_val_14b[:, 0:4] = 0.0
        p14 = self.model_14b.predict(X_val_14b)
        p06 = self.model_06b_rag.predict(X_val)
        benefit = p06 - p14

        # groud_truth: RAG is truly needed when 14B fails (0) and RAG succeeds (1). Otherwise, 0.
        ground_truth = ((y14_val == 0) & (yrag_val == 1)).astype(int)

        best_score = -1.0
        penalty_weight = 0.25  # weight to reduce calls to RAG

        for d in np.linspace(
            0.3, 0.7, 81
        ):  # explore deltas in higher range in order to reduce calls of unnecessary RAG
            route = (p14 < self.trust_threshold) & (benefit > d)
            if np.sum(route) == 0:
                continue

            # Router's selection accuracy
            mcc = matthews_corrcoef(ground_truth, route.astype(int))
            score = mcc - (penalty_weight * np.mean(route))

            if score > best_score:
                best_score = score
                self.best_delta = d

        print(f"Aggressive Delta Optimized: {self.best_delta:.3f}")

    def predict_route(self, X):
        X_14b = X.copy()
        X_14b[:, 0:4] = 0.0
        p14 = self.model_14b.predict(X_14b)
        p06 = self.model_06b_rag.predict(X)
        benefit = p06 - p14
        route_to_rag = (p14 < self.trust_threshold) & (benefit > self.best_delta)
        return route_to_rag, p14, p06


def prepare_datasets(args, extractor):
    jsonl_path = f"data/{args.data_type}/triviaqa::olmes_q_retrieved_results::_IVFPQ.65536.64.256::k5.jsonl"
    qid_list, all_scores = load_scores_and_data(jsonl_path)
    score_map = dict(zip(qid_list, all_scores))
    query_map = json2dict(f"data/{args.data_type}/query_map.json")

    datasets = {}
    for split in ["train", "test"]:
        df = pd.read_csv(f"data/{args.data_type}/{split}_local.csv")
        df["query_id_str"] = df["query_id"].astype(str)

        queries_text = [query_map.get(str(qid), "") for qid in df["query_id"]]
        # using 128-dim to secure expressiveness
        q_pca = PCA(n_components=128).fit_transform(
            extractor.model.encode(queries_text, show_progress_bar=True)
        )

        ce_path = f"data/{args.data_type}/ce_score_top5_{'train_stacking' if split == 'train' else 'all'}.csv"
        ce_df = pd.read_csv(ce_path)
        ce_df["query_id"] = ce_df["query_id"].astype(str)
        merged = pd.merge(
            df, ce_df, left_on="query_id_str", right_on="query_id", how="left"
        )

        ce_scores = merged[[f"ce_score_{i}" for i in range(1, 6)]].fillna(0.0).values
        batch_scores = np.array(
            [score_map.get(qid, [0.0] * 5) for qid in merged["query_id_str"]]
        )

        X = extractor.get_features(
            queries_text, batch_scores, ce_scores, query_emb_pca=q_pca
        )
        y_rag = merged["0"].values.astype(float)
        y_14b = pd.read_csv(f"data/{args.data_type}/{split}.csv")["1"].values.astype(
            float
        )

        datasets[split] = {"X": X, "y_14b": y_14b, "y_rag": y_rag}
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="triviaqa_full")
    parser.add_argument(
        "--trust_14b", type=float, default=0.82
    )  # threshold for trusting 14B model
    args = parser.parse_args()

    extractor = FeatureExtractor()
    data = prepare_datasets(args, extractor)

    router = CascadeRouter(trust_14b_threshold=args.trust_14b)
    router.train(data["train"]["X"], data["train"]["y_14b"], data["train"]["y_rag"])

    preds_is_rag, p14, p06 = router.predict_route(data["test"]["X"])
    y14, yrag = data["test"]["y_14b"], data["test"]["y_rag"]

    # accuracy across the test dataset
    final_res = np.where(preds_is_rag, yrag, y14)
    system_acc = np.mean(final_res)

    # Router's selection correctness
    # when 14B fails (0) and RAG succeeds (1), RAG was truly needed. Otherwise, 14B was sufficient.
    router_target = ((y14 == 0) & (yrag == 1)).astype(int)
    router_preds = preds_is_rag.astype(int)

    r_acc = accuracy_score(router_target, router_preds)
    r_pre = precision_score(router_target, router_preds, zero_division=0)
    r_rec = recall_score(router_target, router_preds, zero_division=0)
    r_mcc = matthews_corrcoef(router_target, router_preds)

    print(f"=== System Performance Evaluation ===")
    print(f"Total System Accuracy: {system_acc:.4f} (Baseline 14B: {np.mean(y14):.4f})")
    print(f"RAG Call Rate: {np.mean(preds_is_rag):.2%}")

    print(f"=== Router Decision Quality (Is RAG really needed?) ===")
    print(f"Router Selection Accuracy: {r_acc:.4f}")
    print(f"├─ Precision (batting accuracy / effective RAG): {r_pre:.4f}")
    print(f"├─ Recall    (coverage / rescue rate): {r_rec:.4f}")
    print(f"└─ MCC       (overall plate discipline): {r_mcc:.4f}")

    cm = confusion_matrix(router_target, router_preds)
    print(f"Confusion Matrix (Selection Strategy):")
    print(f"                Pred: 14B | Pred: RAG")
    print(f"Actual: 14B-OK:  {cm[0,0]:>7} | {cm[0,1]:>9} (Overkill or Waste)")
    print(f"Actual: NeedRAG: {cm[1,0]:>7} | {cm[1,1]:>9} (Saved!)")


if __name__ == "__main__":
    main()
