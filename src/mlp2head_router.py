import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def json2dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_scores_and_data(jsonl_path: str):
    # top 5 assumed
    if not os.path.exists(jsonl_path):
        print(f"[WARN] retrieval jsonl not found: {jsonl_path}")
        return [], {}
    qid_list, score_map = [], {}
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            q_id = str(data.get("id", data.get("question_id")))
            ctxs = data.get("ctxs", [])
            scores = [float(ctx.get("retrieval score", 0)) for ctx in ctxs]
            while len(scores) < 5:
                scores.append(0.0)
            score_map[q_id] = scores[:5]
            qid_list.append(q_id)
    return qid_list, score_map


def _safe_softmax(x, axis=-1, eps=1e-12):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + eps)


def _entropy_from_scores(scores, eps=1e-12):
    # scores: (N,5)
    p = _safe_softmax(scores, axis=1, eps=eps)
    return -(p * np.log(p + eps)).sum(axis=1)


def _to_tensor(x_np, device):
    return torch.from_numpy(x_np).to(device=device, dtype=torch.float32)


def _batches(n, batch_size):
    idx = np.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]

class FeatureExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        self.device = device
        self.st_model = SentenceTransformer(model_name, device=device)

    def encode_queries_np(self, queries):
        with torch.no_grad():
            q_emb = self.st_model.encode(
                queries, convert_to_tensor=True, show_progress_bar=False
            )
        return q_emb.detach().cpu().numpy().astype(np.float32)

    def encode_docs_np(self, docs):
        with torch.no_grad():
            d_emb = self.st_model.encode(
                docs, convert_to_tensor=True, show_progress_bar=False
            )
        return d_emb.detach().cpu().numpy().astype(np.float32)

    def get_features(self, queries, docs, ret_scores_np, ce_scores_top5, docs_top5=None):
        # cosine(q_emb, d_emb) for paired (q, top1doc)
        with torch.no_grad():
            q_emb = self.st_model.encode(
                queries, convert_to_tensor=True, show_progress_bar=False
            )
            d_emb = self.st_model.encode(
                docs, convert_to_tensor=True, show_progress_bar=False
            )
            cos_qd = (
                torch.cosine_similarity(q_emb, d_emb, dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        cos_qd = cos_qd.reshape(-1, 1)

        ce_top1 = ce_scores_top5[:, 0].reshape(-1, 1).astype(np.float32)
        ce_diff = (ce_scores_top5[:, 0] - ce_scores_top5[:, 1]).reshape(-1, 1).astype(
            np.float32
        )

        ret_top1 = ret_scores_np[:, 0].reshape(-1, 1).astype(np.float32)
        ret_gap12 = (
            (ret_scores_np[:, 0] - ret_scores_np[:, 1])
            .reshape(-1, 1)
            .astype(np.float32)
        )
        ret_gap15 = (
            (ret_scores_np[:, 0] - ret_scores_np[:, 4])
            .reshape(-1, 1)
            .astype(np.float32)
        )
        ret_mean = ret_scores_np.mean(axis=1).reshape(-1, 1).astype(np.float32)
        ret_std = ret_scores_np.std(axis=1).reshape(-1, 1).astype(np.float32)
        ret_ent = _entropy_from_scores(ret_scores_np).reshape(-1, 1).astype(np.float32)

        q_len = np.array([len(q) for q in queries], dtype=np.float32).reshape(-1, 1)
        doc_len_top1 = np.array([len(d) for d in docs], dtype=np.float32).reshape(-1, 1)

        if docs_top5 is not None:
            doc_lens_top5 = np.array([[len(d) for d in ds] for ds in docs_top5], dtype=np.float32)
            doc_len_mean = doc_lens_top5.mean(axis=1, keepdims=True)
            doc_len_std = doc_lens_top5.std(axis=1, keepdims=True)
        else:
            doc_len_mean = doc_len_top1
            doc_len_std = np.zeros_like(doc_len_top1)

        feats = [
            cos_qd,
            ce_top1, ce_diff,
            ret_top1, ret_gap12, ret_gap15, ret_mean, ret_std, ret_ent,
            q_len,
            doc_len_top1, doc_len_mean, doc_len_std,
        ]
        return np.hstack(feats).astype(np.float32)


class MLP2Head(nn.Module):
    """
    logits[:,0] -> 14B correct logit
    logits[:,1] -> RAG (0.6B+RAG) correct logit
    """

    def __init__(self, in_dim: int, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must be a tuple like (h1, h2)")
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout = nn.Dropout(dropout)

        self.head_14b = nn.Linear(h2, 1)
        self.head_rag = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logit_14b = self.head_14b(x)
        logit_rag = self.head_rag(x)
        logits = torch.cat([logit_14b, logit_rag], dim=1)  # (N,2)
        return logits

@dataclass
class ValBest:
    val_loss: float
    state_dict: dict


class Router2Prob:
    def __init__(
        self,
        energy_penalty_weight=0.25,  # penalty against calling RAG
        hidden_dims=(64, 32),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=200,
        batch_size=512,
        patience=20,
        pos_weight_14=1.0,
        pos_weight_rag=1.0,
        device="cuda",
    ):
        self.energy_penalty_weight = float(energy_penalty_weight)
        self.hidden_dims = hidden_dims
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.pos_weight_14 = float(pos_weight_14)
        self.pos_weight_rag = float(pos_weight_rag)
        self.device = device

        self.model = None
        self.scaler = None
        self.best_delta = 0.0
        self.feature_names = None

    def train(self, X_train: np.ndarray, y14: np.ndarray, yrag: np.ndarray, feature_names):
        """
        y14, yrag are {0,1} float or int arrays.
        """
        self.feature_names = list(feature_names)

        Y = np.stack([y14, yrag], axis=1).astype(np.float32)  # (N,2)

        X_t, X_v, Y_t, Y_v = train_test_split(
            X_train, Y, test_size=0.15, stratify=Y[:, 1].astype(int), random_state=42
        )

        in_dim = X_t.shape[1]
        self.model = MLP2Head(
            in_dim=in_dim, hidden_dims=self.hidden_dims, dropout=self.dropout
        ).to(self.device)

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        pos_w = torch.tensor([self.pos_weight_14, self.pos_weight_rag], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        X_v_t = _to_tensor(X_v, self.device)
        Y_v_t = _to_tensor(Y_v, self.device)

        best = ValBest(val_loss=float("inf"), state_dict=None)
        bad = 0

        print("Training Router2Prob (predict p(14B correct) and p(RAG correct))...")
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            perm = np.random.permutation(len(X_t))
            X_t_shuf = X_t[perm]
            Y_t_shuf = Y_t[perm]

            total_loss = 0.0
            for bidx in _batches(len(X_t_shuf), self.batch_size):
                xb = _to_tensor(X_t_shuf[bidx], self.device)
                yb = _to_tensor(Y_t_shuf[bidx], self.device)  # (B,2)

                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)  # (B,2)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                total_loss += float(loss.item()) * len(bidx)

            self.model.eval()
            with torch.no_grad():
                v_logits = self.model(X_v_t)
                v_loss = float(criterion(v_logits, Y_v_t).item())

            avg_train = total_loss / len(X_t_shuf)
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | train_loss={avg_train:.5f} | val_loss={v_loss:.5f}")

            if v_loss < best.val_loss - 1e-5:
                best.val_loss = v_loss
                best.state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best val_loss={best.val_loss:.5f})")
                    break

        if best.state_dict is not None:
            self.model.load_state_dict(best.state_dict)

        self._optimize_delta(X_v, Y_v[:, 0], Y_v[:, 1])

    def predict_probs(self, X: np.ndarray):
        """
        Returns p14, prag as numpy arrays shape (N,)
        """
        self.model.eval()
        with torch.no_grad():
            xt = _to_tensor(X, self.device)
            logits = self.model(xt)  # (N,2)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        p14 = p[:, 0]
        prag = p[:, 1]
        return p14, prag

    def _route_from_probs(self, p14: np.ndarray, prag: np.ndarray, delta: float):
        return (prag - p14) > float(delta)

    def _optimize_delta(self, X_val: np.ndarray, y14_val: np.ndarray, yrag_val: np.ndarray):
        """
        Choose delta to maximize:
          score = system_accuracy - energy_penalty_weight * rag_call_rate

        Where system_accuracy uses *ground-truth* correctness for the selected option:
          final_correct = yrag if route_rag else y14
        """
        print(f"Optimizing delta with score = acc - penalty*call_rate (penalty={self.energy_penalty_weight})")

        p14, prag = self.predict_probs(X_val)

        best_score = -1e9
        best_delta = 0.0
        best_acc = 0.0
        best_call = 0.0

        # Reasonable sweep range. You can widen if needed.
        for delta in np.linspace(-0.5, 0.5, 401):
            route_rag = self._route_from_probs(p14, prag, delta)
            call_rate = float(np.mean(route_rag))
            final_correct = np.where(route_rag, yrag_val, y14_val)
            acc = float(np.mean(final_correct))

            score = acc - self.energy_penalty_weight * call_rate
            if score > best_score:
                best_score = score
                best_delta = float(delta)
                best_acc = acc
                best_call = call_rate

        self.best_delta = best_delta
        print(f"Best delta: {self.best_delta:+.4f} | acc={best_acc:.4f} | call_rate={best_call:.2%} | score={best_score:.4f}")

    def predict_route(self, X: np.ndarray):
        """
        Returns (route_rag_bool, p14, prag)
        """
        p14, prag = self.predict_probs(X)
        route_rag = self._route_from_probs(p14, prag, self.best_delta)
        return route_rag.astype(bool), p14, prag
    
import math

def _mcc_from_confusion(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="triviaqa_full")
    parser.add_argument("--penalty", type=float, default=0.25)
    parser.add_argument("--hidden", type=str, default="64,32")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--pos_weight_14", type=float, default=1.0)
    parser.add_argument("--pos_weight_rag", type=float, default=1.0)
    parser.add_argument("--pca_dim", type=int, default=32, help="PCA dim for query embedding")
    parser.add_argument("--pca_d_dim", type=int, default=8, help="PCA dim for doc (top1) embedding")
    parser.add_argument("--no_scale", action="store_true")
    args = parser.parse_args()

    base = f"data/{args.data_type}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(device=device)

    jsonl_path = f"{base}/triviaqa::olmes_q_retrieved_results::_IVFPQ.65536.64.256::k5.jsonl"
    _, score_map = load_scores_and_data(jsonl_path)

    query_map = json2dict(f"{base}/query_map.json")
    doc_map = json2dict(f"{base}/doc_map.json")

    f_names_base = [
        "cos_qd",
        "ce_top1", "ce_diff",
        "ret_top1", "ret_gap12", "ret_gap15",
        "ret_mean", "ret_std", "ret_entropy",
        "q_len",
        "doc_len_top1", "doc_len_mean", "doc_len_std",
    ]

    datasets = {}
    for split in ["train", "test"]:
        print(f"Processing {split} features...")
        df = pd.read_csv(f"{base}/{split}_local.csv")
        df["query_id_str"] = df["query_id"].astype(str)

        ce_path = f"{base}/ce_score_top5_{'train_stacking' if split == 'train' else 'all'}.csv"
        ce_df = pd.read_csv(ce_path)
        ce_df["query_id"] = ce_df["query_id"].astype(str)
        merged = pd.merge(df, ce_df, left_on="query_id_str", right_on="query_id", how="left").fillna(0)

        q_texts = [query_map.get(str(qid), "") for qid in merged["query_id_str"]]
        d_texts = [doc_map.get(str(did), "") for did in merged["doc_id"]]

        ce_scores = merged[[f"ce_score_{i}" for i in range(1, 6)]].values.astype(np.float32)
        ret_scores = np.array([score_map.get(qid, [0.0] * 5) for qid in merged["query_id_str"]], dtype=np.float32)

        X_base = extractor.get_features(q_texts, d_texts, ret_scores, ce_scores, docs_top5=None)

        q_emb = extractor.encode_queries_np(q_texts)  # (N, emb_dim)
        d_emb = extractor.encode_docs_np(d_texts)     # (N, emb_dim)

        y_rag = merged["0"].values.astype(np.float32)  # 0.6B+RAG correctness
        y_14b = pd.read_csv(f"{base}/{split}.csv")["1"].values.astype(np.float32)  # 14B correctness

        datasets[split] = {
            "X_base": X_base,
            "q_emb": q_emb,
            "d_emb": d_emb,
            "y14": y_14b,
            "yrag": y_rag,
        }

        if split == "train":
            only_rag = ((y_rag == 1.0) & (y_14b == 0.0)).astype(np.float32)
            only_14 = ((y_14b == 1.0) & (y_rag == 0.0)).astype(np.float32)
            print(f"[train] only-RAG correct rate: {only_rag.mean():.4f}")
            print(f"[train] only-14B correct rate: {only_14.mean():.4f}")

    f_names = list(f_names_base)
    pca_dim = int(args.pca_dim)
    if pca_dim > 0:
        pca_q = PCA(n_components=pca_dim, random_state=42)
        q_train_pca = pca_q.fit_transform(datasets["train"]["q_emb"]).astype(np.float32)
        q_test_pca = pca_q.transform(datasets["test"]["q_emb"]).astype(np.float32)
        f_names += [f"q_pca_{i}" for i in range(pca_dim)]
    else:
        q_train_pca = None
        q_test_pca = None

    pca_d_dim = int(args.pca_d_dim)
    if pca_d_dim > 0:
        pca_d = PCA(n_components=pca_d_dim, random_state=42)
        d_train_pca = pca_d.fit_transform(datasets["train"]["d_emb"]).astype(np.float32)
        d_test_pca = pca_d.transform(datasets["test"]["d_emb"]).astype(np.float32)
        f_names += [f"d_pca_{i}" for i in range(pca_d_dim)]
    else:
        d_train_pca = None
        d_test_pca = None

    X_train = datasets["train"]["X_base"]
    X_test = datasets["test"]["X_base"]

    if q_train_pca is not None:
        X_train = np.hstack([X_train, q_train_pca]).astype(np.float32)
        X_test = np.hstack([X_test, q_test_pca]).astype(np.float32)

    if d_train_pca is not None:
        X_train = np.hstack([X_train, d_train_pca]).astype(np.float32)
        X_test = np.hstack([X_test, d_test_pca]).astype(np.float32)

    if not args.no_scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    hidden_dims = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    if len(hidden_dims) != 2:
        raise ValueError("--hidden must be like '64,32'")

    router = Router2Prob(
        energy_penalty_weight=args.penalty,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        pos_weight_14=args.pos_weight_14,
        pos_weight_rag=args.pos_weight_rag,
        device=device,
    )

    y14_train = datasets["train"]["y14"].astype(np.float32)
    yrag_train = datasets["train"]["yrag"].astype(np.float32)
    router.train(X_train, y14_train, yrag_train, f_names)

    route_rag, p14, prag = router.predict_route(X_test)

    y14_test = datasets["test"]["y14"].astype(np.float32)
    yrag_test = datasets["test"]["yrag"].astype(np.float32)

    final_correct = np.where(route_rag, yrag_test, y14_test)
    base14_correct = y14_test

    actual_only_14 = (y14_test == 1.0) & (yrag_test == 0.0)
    actual_only_rag = (yrag_test == 1.0) & (y14_test == 0.0)
    both_correct = (yrag_test == 1.0) & (y14_test == 1.0)
    both_wrong = (yrag_test == 0.0) & (y14_test == 0.0)

    # Router decision quality wrt "only-RAG correct"
    y_actual_only_rag = actual_only_rag.astype(int)
    y_pred_route = route_rag.astype(int)

    # simple metrics w/o sklearn (to keep deps minimal)
    tp = int(np.sum((y_pred_route == 1) & (y_actual_only_rag == 1)))
    tn = int(np.sum((y_pred_route == 0) & (y_actual_only_rag == 0)))
    fp = int(np.sum((y_pred_route == 1) & (y_actual_only_rag == 0)))
    fn = int(np.sum((y_pred_route == 0) & (y_actual_only_rag == 1)))

    sel_acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    mcc  = _mcc_from_confusion(tp, tn, fp, fn)

    print("=== System Performance Evaluation ===")
    print(f"Best delta (val-optimized): {router.best_delta:+.4f}")
    print(f"Total System Accuracy: {float(np.mean(final_correct)):.4f} (Baseline 14B: {float(np.mean(base14_correct)):.4f})")
    print(f"RAG Call Rate: {float(np.mean(route_rag)):.2%}")

    print("=== Predicted Probability Sanity (test) ===")
    print(f"mean p14={float(np.mean(p14)):.4f} | mean prag={float(np.mean(prag)):.4f} | mean (prag-p14)={float(np.mean(prag-p14)):.4f}")

    print("=== Router Decision Quality (label = only-RAG-correct) ===")
    print(f"Selection Accuracy: {sel_acc:.4f}")
    print(f"├─ Precision (Route to RAG when only-RAG): {prec:.4f}")
    print(f"└─ Recall    (Capture only-RAG cases):     {rec:.4f}")
    print(f"└─ MCC       (Overall routing quality):    {mcc:.4f}")
    print("Confusion Matrix (Routing Decision)")
    print("                    Pred: Choose 14B | Pred: Choose RAG")
    print(f"Actual: Not-only-RAG     {tn:>10} | {fp:>13}  (Potential wasted call)")
    print(f"Actual: Only-RAG         {fn:>10} | {tp:>13}  (Saved!)")

    print("=== Outcome Breakdown (test) ===")
    print(f"Actual 14B (14B correct, RAG wrong): {actual_only_14.sum():>7} ({actual_only_14.mean():.2%})")
    print(f"Actual RAG (RAG correct, 14B wrong): {actual_only_rag.sum():>7} ({actual_only_rag.mean():.2%})")
    print(f"Both correct:                            {both_correct.sum():>7} ({both_correct.mean():.2%})")
    print(f"Both wrong:                             {both_wrong.sum():>7} ({both_wrong.sum()/len(y14_test):.2%})")

    with torch.no_grad():
        first_linear = None
        for m in router.model.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is not None and router.feature_names is not None:
            w = first_linear.weight.detach().cpu().numpy()  # (h1, in_dim)
            approx_imp = np.linalg.norm(w, axis=0)         # (in_dim,)
            imp = pd.DataFrame({"f": router.feature_names, "v": approx_imp})
            print("--- Approx Feature Importance (First Layer Weight Norm) ---")
            print(imp.sort_values("v", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
