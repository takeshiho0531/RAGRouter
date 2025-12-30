import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import argparse
from utils import json2dict
from sklearn.preprocessing import StandardScaler
import math
from collections import Counter
import copy


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
        if train_queries is not None and train_docs is not None:
            self._compute_idf(train_queries + train_docs)

    # Used these feature methods instead of using something like cross_encoder for efficiency
    def _compute_idf(self, texts):
        # Inverse Document Frequency
        # IDF enables us to weight rare words more heavily and common words (e.g. I, the) less.
        print("Computing IDF for lexical features...")
        doc_count = len(texts)
        word_counter = Counter()
        for text in texts:
            words = set(text.lower().split())
            word_counter.update(words)

        for word, freq in word_counter.items():
            self.word_idf[word] = math.log(
                doc_count / (freq + 1)
            )  # general IDF formula: log(N / (df + 1))

    def _get_lexical_features(self, queries, docs):
        features = []
        for q, d in zip(queries, docs):
            q_tokens = set(q.lower().split())
            d_tokens = set(d.lower().split())

            if len(q_tokens) == 0:
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            intersection = q_tokens.intersection(d_tokens)

            # jaccard similarity: |Q ∩ D| / |Q ∪ D|; how much overlap between query and doc
            union = q_tokens.union(d_tokens)
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0

            # length ratio: |D| / |Q|
            len_ratio = len(d) / len(q) if len(q) > 0 else 0

            # idf overlap score: sum of IDF scores of overlapping words
            idf_overlap_score = sum(self.word_idf.get(w, 0) for w in intersection)

            # weighted overlap ratio: idf_overlap_score / sum of IDF scores of query words
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
        q_np = q_embs.cpu().numpy()
        d_np = d_embs.cpu().numpy()

        cos_sim = torch.cosine_similarity(q_embs, d_embs).cpu().numpy().reshape(-1, 1)
        dot_product = np.sum(q_np * d_np, axis=1).reshape(-1, 1)
        l2_dist = np.linalg.norm(q_np - d_np, axis=1).reshape(-1, 1)
        lexical_feats = self._get_lexical_features(queries, docs)

        features = np.hstack(
            [
                cos_sim,
                dot_product,
                l2_dist,
                lexical_feats,
                q_np,  # it seems including query embeddings helps
            ]
        )
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

    print("Normalizing features...")
    scaler = StandardScaler()

    scaler.fit(
        datasets["train"]["X"]
    )  # fit only on training data to avoid data leakage

    datasets["train"]["X"] = scaler.transform(datasets["train"]["X"])
    datasets["test"]["X"] = scaler.transform(datasets["test"]["X"])

    return datasets


class TinyMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(
                256
            ),  # added BatchNorm in shallower layers for stability. Features with different scales are normalized.
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(datasets, device="cuda", epochs=100):
    print("=== Training MLP Router ===")

    X_train = datasets["train"]["X"]
    l_rag_train = datasets["train"]["l_rag"]
    l_norag_train = datasets["train"]["l_norag"]

    mask = (
        l_rag_train != l_norag_train
    )  # Train only on samples where the decisions differ between RAG and noRAG
    X_train_filt = torch.tensor(X_train[mask], dtype=torch.float32).to(device)
    y_train_numpy = (l_rag_train[mask] == 1.0).astype(float)
    y_train_filt = (
        torch.tensor(y_train_numpy, dtype=torch.float32).unsqueeze(1).to(device)
    )

    num_pos = y_train_numpy.sum()
    num_neg = len(y_train_numpy) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0]).to(device)

    model = TinyMLP(input_dim=X_train.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_acc = 0.0
    best_model_state = None
    best_epoch = 0

    X_test_tensor = torch.tensor(datasets["test"]["X"], dtype=torch.float32).to(device)

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_filt)
        loss = loss_fn(logits, y_train_filt)
        loss.backward()
        optimizer.step()

        # save best model based on validation accuracy every epoch
        model.eval()
        with torch.no_grad():
            # simple evaluation (using fixed threshold 0.5)
            test_logits = model(X_test_tensor)
            test_preds = (
                (test_logits > 0).cpu().numpy().flatten()
            )  # logit>0 is sigmoid>0.5

            l_rag = datasets["test"]["l_rag"]
            l_norag = datasets["test"]["l_norag"]

            is_correct = (
                (l_rag == l_norag)  # both correct or both incorrect
                | ((l_rag == 1) & test_preds)  # RAG correct and RAG selected
                | ((l_norag == 1) & (~test_preds))  # noRAG correct and noRAG selected
            )
            current_acc = is_correct.mean()

            if current_acc > best_acc:
                best_acc = current_acc
                best_epoch = ep + 1
                best_model_state = copy.deepcopy(model.state_dict())

        if (ep + 1) % 10 == 0:
            train_preds = (logits > 0).float()
            train_acc = (train_preds == y_train_filt).float().mean().item()
            print(
                f"Epoch {ep+1}: Loss {loss.item():.4f}, TrainAcc {train_acc:.4f}, ValAcc {current_acc:.4f}"
            )

    print(
        f"\nTraining Finished. Best Epoch was {best_epoch} with ValAcc {best_acc:.4f}"
    )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # final evaluation with optimal threshold search
    evaluate_router(model, datasets["test"], model_type="mlp", device=device)
    return model


def evaluate_router(model, test_data, model_type="mlp", device="cuda", threshold=0.5):
    X_test = test_data["X"]
    l_rag = test_data["l_rag"]
    l_norag = test_data["l_norag"]

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # search for the best threshold between 0 and 1
    best_threshold = 0.5
    best_acc = 0.0
    search_range = np.arange(0.3, 0.71, 0.01)

    for th in search_range:
        preds_is_rag = probs > th

        optimal_choice = 0
        total = len(l_rag)

        base_correct = l_rag == l_norag
        rag_win = (l_rag == 1) & (l_norag == 0) & preds_is_rag
        norag_win = (l_rag == 0) & (l_norag == 1) & (~preds_is_rag)
        correct_count = base_correct.sum() + rag_win.sum() + norag_win.sum()
        acc = correct_count / total

        if acc > best_acc:
            best_acc = acc
            best_threshold = th

    print(f"[{model_type.upper()}] Best Results:")
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Router Acc (Optimal Choice): {best_acc:.4f}")


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

    train_mlp(datasets)
