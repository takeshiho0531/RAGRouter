import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import argparse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


def json2dict(path):
    with open(path, "r") as f:
        return json.load(f)


class FeatureExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)

    def get_features(self, queries, q_pca=None):
        q_len = np.array([len(q) for q in queries]).reshape(-1, 1)
        q_words = np.array([len(q.split()) for q in queries]).reshape(-1, 1)
        features = [q_len, q_words]
        if q_pca is not None:
            features.append(q_pca)
        return np.hstack(features)


class SuccessPredictor14B:
    def __init__(self):
        self.model = None
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        }

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        ds_train = lgb.Dataset(X_train, label=y_train)
        ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

        print("Training 14B Success Predictor...")
        self.model = lgb.train(
            self.params,
            ds_train,
            num_boost_round=2000,
            valid_sets=[ds_val],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100),
            ],
        )

    def predict_proba(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)


def prepare_datasets(args, extractor, n_components=32):
    query_map = json2dict(f"data/{args.data_type}/query_map.json")

    pca_q = PCA(n_components=n_components)
    raw_data = {}

    for split in ["train", "test"]:
        df = pd.read_csv(f"data/{args.data_type}/{split}_local.csv")
        y_14b = pd.read_csv(f"data/{args.data_type}/{split}.csv")["1"].values.astype(
            float
        )

        q_text = [query_map.get(str(qid), "") for qid in df["query_id"]]
        q_emb = extractor.model.encode(q_text, show_progress_bar=True)
        raw_data[split] = {"q_text": q_text, "q_emb": q_emb, "y": y_14b}

    print(f"Fitting PCA (dim={n_components}) on Train query embeddings...")
    pca_q.fit(raw_data["train"]["q_emb"])

    datasets = {}
    for split in ["train", "test"]:
        q_pca = pca_q.transform(raw_data[split]["q_emb"])
        X = extractor.get_features(raw_data[split]["q_text"], q_pca=q_pca)
        datasets[split] = {"X": X.astype(np.float32), "y": raw_data[split]["y"]}

    f_names = ["q_len", "q_words"] + [f"q_pca_{i}" for i in range(n_components)]
    return datasets, f_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="triviaqa_full")
    parser.add_argument("--pca_dim", type=int, default=32)  # able to set PCA dimension
    args = parser.parse_args()

    extractor = FeatureExtractor()
    data, f_names = prepare_datasets(args, extractor, n_components=args.pca_dim)

    predictor = SuccessPredictor14B()
    predictor.train(data["train"]["X"], data["train"]["y"])

    probs = predictor.predict_proba(data["test"]["X"])
    preds = (probs > 0.5).astype(int)
    y_true = data["test"]["y"]

    print("\n" + "=" * 40)
    print("  14B Success Prediction Results")
    print("=" * 40)
    print(f"Accuracy:  {accuracy_score(y_true, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, probs):.4f}")
    print(f"MCC:       {matthews_corrcoef(y_true, preds):.4f}")

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, preds)
    print(f"                Pred: Fail | Pred: Success")
    print(f"Actual: Fail:    {cm[0,0]:>9} | {cm[0,1]:>12}")
    print(f"Actual: Success: {cm[1,0]:>9} | {cm[1,1]:>12}")

    print("--- Top 10 Important Features ---")
    importances = predictor.model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": f_names, "importance": importances})
    print(imp_df.sort_values("importance", ascending=False).head(10))


if __name__ == "__main__":
    main()
