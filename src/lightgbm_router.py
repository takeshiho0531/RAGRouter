import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import argparse
from utils import json2dict

import math
from collections import Counter


class FeatureExtractor:
    # reference: mlp_router.py
    def __init__(
        self,
        train_queries=None,
        train_docs=None,
        model_name="all-MiniLM-L6-v2",
        device="cuda",
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.word_idf = {}
        if train_queries is not None and train_docs is not None:
            self._compute_idf(train_queries + train_docs)

    def _compute_idf(self, texts):
        print("Computing IDF for lexical features...")
        doc_count = len(texts)
        word_counter = Counter()
        for text in texts:
            words = set(text.lower().split())
            word_counter.update(words)

        for word, freq in word_counter.items():
            self.word_idf[word] = math.log(doc_count / (freq + 1))

    def _get_lexical_features(self, queries, docs):
        features = []
        for q, d in zip(queries, docs):
            q_tokens = set(q.lower().split())
            d_tokens = set(d.lower().split())

            if len(q_tokens) == 0:
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            intersection = q_tokens.intersection(d_tokens)

            union = q_tokens.union(d_tokens)
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0

            len_ratio = len(d) / len(q) if len(q) > 0 else 0

            idf_overlap_score = sum(self.word_idf.get(w, 0) for w in intersection)

            q_total_idf = sum(self.word_idf.get(w, 0) for w in q_tokens)
            weighted_overlap_ratio = (
                idf_overlap_score / q_total_idf if q_total_idf > 0 else 0
            )

            features.append(
                [jaccard, len_ratio, idf_overlap_score, weighted_overlap_ratio]
            )

        return np.array(features)

    def get_features(self, queries, docs):
        q_embs = self.model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        )
        d_embs = self.model.encode(
            docs, convert_to_tensor=True, show_progress_bar=False
        )
        scores = torch.cosine_similarity(q_embs, d_embs).cpu().numpy()
        q_embs_np = q_embs.cpu().numpy()

        lexical_feats = self._get_lexical_features(queries, docs)
        features = np.hstack([scores.reshape(-1, 1), lexical_feats, q_embs_np])
        return features


def prepare_data(args, extractor):
    query_map = json2dict(f"data/{args.data_type}/query_map.json")
    doc_map = json2dict(f"data/{args.data_type}/doc_map.json")

    datasets = {}
    for split in ["train", "test"]:
        rag_df = pd.read_csv(f"data/{args.data_type}/{split}_{args.task_type}.csv")
        norag_df = pd.read_csv(f"data/{args.data_type}/{split}.csv")

        queries = [query_map.get(str(qid), "") for qid in rag_df["query_id"]]
        docs = [doc_map.get(str(did), "") for did in rag_df["doc_id"]]

        l_rag = rag_df["0"].values.astype(float)
        l_norag = norag_df["1"].values.astype(float)

        print(f"Extracting features for {split}...")
        X = extractor.get_features(queries, docs)

        datasets[split] = {"X": X, "l_rag": l_rag, "l_norag": l_norag}
    return datasets


def train_lightgbm(datasets):
    print("=== Training LightGBM Router ===")

    X_train = datasets["train"]["X"]
    l_rag_train = datasets["train"]["l_rag"]
    l_norag_train = datasets["train"]["l_norag"]

    # train only conflict cases; rag is correct and norag is wrong, or vice versa
    mask = l_rag_train != l_norag_train
    X_train_filtered = X_train[mask]
    y_train_filtered = (l_rag_train[mask] == 1.0).astype(int)

    print(f"Training samples (conflict cases only): {len(X_train_filtered)}")

    train_data = lgb.Dataset(X_train_filtered, label=y_train_filtered)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "extra_trees": True,
        "num_leaves": 31,
        "lambda_l1": 5.0,
        "lambda_l2": 5.0,
        "learning_rate": 0.005,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 100,
        "verbose": -1,
    }

    gbm = lgb.train(params, train_data, num_boost_round=2000)

    print("Searching for best threshold...")
    y_pred_prob = gbm.predict(X_train_filtered)

    best_thr = 0.5
    best_acc = 0.0

    for thr in np.arange(0.1, 0.9, 0.001):
        preds = (y_pred_prob > thr).astype(int)
        acc = accuracy_score(y_train_filtered, preds)  # accuracy on conflict cases

        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    print(
        f"Best Threshold Found: {best_thr:.3f} (Conflict-only Train Acc: {best_acc:.4f})"
    )

    print("=== train data evaluation ===")
    evaluate_router(gbm, datasets["train"], model_type="lightgbm", threshold=best_thr)
    print("=== test data evaluation ===")
    evaluate_router(gbm, datasets["test"], model_type="lightgbm", threshold=best_thr)
    return gbm


def evaluate_router(
    model, test_data, model_type="lightgbm", device="cuda", threshold=0.5
):
    X_test = test_data["X"]
    l_rag = test_data["l_rag"]
    l_norag = test_data["l_norag"]

    probs = model.predict(X_test)
    preds_is_rag = probs > threshold

    total_samples = len(l_rag)
    router_optimal_choice = 0

    for i in range(total_samples):
        rag_correct = l_rag[i] == 1.0
        norag_correct = l_norag[i] == 1.0
        chose_rag = preds_is_rag[i]

        if rag_correct == norag_correct:  # when both are correct or both are wrong
            router_optimal_choice += 1
        else:
            if rag_correct and chose_rag:
                router_optimal_choice += 1
            elif norag_correct and not chose_rag:
                router_optimal_choice += 1

    optimal_acc = router_optimal_choice / total_samples

    print(f"[{model_type.upper()}] Results (Threshold: {threshold:.3f}):")
    print(f"  Router Acc (Optimal Choice): {optimal_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="webq")
    parser.add_argument("--task_type", type=str, default="local")
    args = parser.parse_args()

    query_map = json2dict(f"data/{args.data_type}/query_map.json")
    doc_map = json2dict(f"data/{args.data_type}/doc_map.json")
    all_texts = list(query_map.values()) + list(doc_map.values())
    extractor = FeatureExtractor(
        train_queries=list(query_map.values()),
        train_docs=list(doc_map.values()),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    datasets = prepare_data(args, extractor)
    train_lightgbm(datasets)
