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

PCA_Q_DIM = 32
PCA_D_DIM = 16


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

    def get_features(self, queries, scores_np, ce_scores_top5, q_pca=None, d_pca=None):
        ce_top1 = ce_scores_top5[:, 0].reshape(-1, 1)
        ce_max = np.max(ce_scores_top5, axis=1).reshape(-1, 1)
        ce_diff = (ce_scores_top5[:, 0] - ce_scores_top5[:, 1]).reshape(-1, 1)
        ret_diff = (scores_np[:, 0] - scores_np[:, 1]).reshape(-1, 1)

        q_len = np.array([len(q) for q in queries]).reshape(-1, 1)
        q_words = np.array([len(q.split()) for q in queries]).reshape(-1, 1)
        agreement = (scores_np[:, 0].reshape(-1, 1)) * ce_top1

        features = [ce_top1, ce_max, ce_diff, ret_diff, q_len, q_words, agreement]
        if q_pca is not None:
            features.append(q_pca)
        if d_pca is not None:
            features.append(d_pca)

        return np.hstack(features)


class CascadeRouter:
    def __init__(self, trust_14b_threshold=0.82):
        self.model_14b = None
        self.model_06b_rag = None
        self.trust_threshold = trust_14b_threshold
        self.best_delta = 0.0
        self.feature_names = None
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.02,
            "feature_fraction": 0.8,
            "verbose": -1,
            "n_jobs": -1,
        }

    def train(self, X_train, y_14b, y_rag, feature_names):
        self.feature_names = feature_names

        retrieval_idx = [0, 1, 2, 3, 6]
        doc_start_idx = 7 + PCA_Q_DIM

        X_train_noise = X_train.copy()
        X_train_noise[:, retrieval_idx] += np.random.normal(
            0, 0.01, X_train_noise[:, retrieval_idx].shape
        )

        X_train_for_14b = X_train_noise.copy()
        # masking retrieval knowledge(0,1,2,3,6) and document knowledge(135~)
        X_train_for_14b[:, retrieval_idx] = 0.0
        X_train_for_14b[:, doc_start_idx:] = 0.0

        X_train_for_rag = X_train_noise.copy()

        (X14_t, X14_v, XR_t, XR_v, y14_t, y14_v, yrag_t, yrag_v) = train_test_split(
            X_train_for_14b,
            X_train_for_rag,
            y_14b,
            y_rag,
            test_size=0.15,
            random_state=42,
        )

        ds_14b_train = lgb.Dataset(X14_t, label=y14_t)
        ds_14b_valid = lgb.Dataset(X14_v, label=y14_v, reference=ds_14b_train)

        print(
            "Training 14B Success Predictor (Zero-Knowledge of Retrieval, Early Stopping enabled)..."
        )
        self.model_14b = lgb.train(
            self.params,
            ds_14b_train,
            num_boost_round=5000,  # 多めに設定
            valid_sets=[ds_14b_valid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),  # 50回改善しなければ終了
                lgb.log_evaluation(period=50),  # 100回ごとにログ出力
            ],
        )

        print(
            "Training 0.6B+RAG Success Predictor (Full-Knowledge, Early Stopping enabled)..."
        )
        weights_t = np.where((y14_t == 0) & (yrag_t == 1), 15.0, 1.0)
        weights_v = np.where((y14_v == 0) & (yrag_v == 1), 15.0, 1.0)

        ds_rag_train = lgb.Dataset(XR_t, label=yrag_t, weight=weights_t)
        ds_rag_valid = lgb.Dataset(
            XR_v, label=yrag_v, weight=weights_v, reference=ds_rag_train
        )

        self.model_06b_rag = lgb.train(
            self.params,
            ds_rag_train,
            num_boost_round=5000,
            valid_sets=[ds_rag_valid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )

        self._optimize_cascade(X14_v, XR_v, y14_v, yrag_v)

    def _optimize_cascade(self, X14_v, XR_v, y14_v, yrag_v):
        print("Optimizing Delta (MCC - Penalty)...")
        p14 = self.model_14b.predict(X14_v, num_iteration=self.model_14b.best_iteration)
        p06 = self.model_06b_rag.predict(
            XR_v, num_iteration=self.model_06b_rag.best_iteration
        )

        benefit = p06 - p14
        ground_truth = ((y14_v == 0) & (yrag_v == 1)).astype(int)

        best_score = -1.0
        energy_penalty_weight = 0.15
        print(f"energy_penalty_weight: {energy_penalty_weight}")
        for d in np.linspace(0.3, 0.7, 81):
            route = (p14 < self.trust_threshold) & (benefit > d)
            if np.sum(route) == 0:
                continue

            score = matthews_corrcoef(ground_truth, route.astype(int)) - (
                energy_penalty_weight * np.mean(route)
            )
            if score > best_score:
                best_score = score
                self.best_delta = d
        print(f"Aggressive Delta Optimized: {self.best_delta:.3f}")

    def predict_route(self, X):
        X_14b = X.copy()
        retrieval_idx = [0, 1, 2, 3, 6]
        doc_start_idx = 7 + PCA_Q_DIM
        X_14b[:, retrieval_idx] = 0.0
        X_14b[:, doc_start_idx:] = 0.0
        p14 = self.model_14b.predict(X_14b, num_iteration=self.model_14b.best_iteration)
        p06 = self.model_06b_rag.predict(
            X, num_iteration=self.model_06b_rag.best_iteration
        )
        route = (p14 < self.trust_threshold) & (p06 - p14 > self.best_delta)
        return route, p14, p06

    def show_importance(self):
        for model_name, model in [
            ("14B", self.model_14b),
            ("0.6B+RAG", self.model_06b_rag),
        ]:
            print(f"--- Feature Importance ({model_name}) Top 10 ---")
            imp = pd.DataFrame(
                {
                    "f": self.feature_names,
                    "v": model.feature_importance(importance_type="gain"),
                }
            )
            print(imp.sort_values("v", ascending=False).head(10))


def prepare_datasets(args, extractor):
    jsonl_path = f"data/{args.data_type}/triviaqa::olmes_q_retrieved_results::_IVFPQ.65536.64.256::k5.jsonl"
    qid_list, all_scores = load_scores_and_data(jsonl_path)
    score_map = dict(zip(qid_list, all_scores))
    query_map = json2dict(f"data/{args.data_type}/query_map.json")
    doc_map = json2dict(f"data/{args.data_type}/doc_map.json")

    pca_q = PCA(n_components=PCA_Q_DIM)
    pca_d = PCA(n_components=PCA_D_DIM)

    raw_data = {}
    for split in ["train", "test"]:
        df = pd.read_csv(f"data/{args.data_type}/{split}_local.csv")
        df["query_id_str"] = df["query_id"].astype(str)

        q_text = [query_map.get(str(qid), "") for qid in df["query_id"]]
        d_text = [doc_map.get(str(did), "") for did in df["doc_id"]]

        q_emb = extractor.model.encode(q_text, show_progress_bar=True)
        d_emb = extractor.model.encode(d_text, show_progress_bar=True)

        raw_data[split] = {"df": df, "q_text": q_text, "q_emb": q_emb, "d_emb": d_emb}

    print("Fitting PCA on Train data only...")
    pca_q.fit(raw_data["train"]["q_emb"])
    pca_d.fit(raw_data["train"]["d_emb"])

    datasets = {}
    for split in ["train", "test"]:
        split_data = raw_data[split]
        df = split_data["df"]

        q_pca = pca_q.transform(split_data["q_emb"])
        d_pca = pca_d.transform(split_data["d_emb"])

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
            split_data["q_text"], batch_scores, ce_scores, q_pca=q_pca, d_pca=d_pca
        )
        X = X.astype(np.float32)
        y_rag = merged["0"].values.astype(float)
        y_14b = pd.read_csv(f"data/{args.data_type}/{split}.csv")["1"].values.astype(
            float
        )
        datasets[split] = {"X": X, "y_14b": y_14b, "y_rag": y_rag}

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="triviaqa_full")
    parser.add_argument(
        "--trust_14b", type=float, default=0.82
    )  # threshold for trusting 14B model
    args = parser.parse_args()

    extractor = FeatureExtractor()
    data, f_names = prepare_datasets(args, extractor)

    router = CascadeRouter(trust_14b_threshold=args.trust_14b)
    router.train(
        data["train"]["X"], data["train"]["y_14b"], data["train"]["y_rag"], f_names
    )

    preds, p14, p06 = router.predict_route(data["test"]["X"])
    y14, yrag = data["test"]["y_14b"], data["test"]["y_rag"]

    # accuracy across the test dataset
    final_res = np.where(preds, yrag, y14)
    system_acc = np.mean(final_res)
    call_rate = np.mean(preds)

    router_target = ((y14 == 0) & (yrag == 1)).astype(int)
    router_preds = preds.astype(int)

    r_acc = accuracy_score(router_target, router_preds)
    r_pre = precision_score(router_target, router_preds, zero_division=0)
    r_rec = recall_score(router_target, router_preds, zero_division=0)
    r_mcc = matthews_corrcoef(router_target, router_preds)

    print(f"=== System Performance Evaluation ===")
    print(f"Total System Accuracy: {system_acc:.4f} (Baseline 14B: {np.mean(y14):.4f})")
    print(f"RAG Call Rate: {call_rate:.2%}")

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

    router.show_importance()


if __name__ == "__main__":
    main()
