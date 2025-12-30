import torch
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from utils import json2dict


class CEFoldDataset(Dataset):
    def __init__(self, df, q_map, d_map, tokenizer, max_len=512):
        self.df = df
        self.q_map = q_map
        self.d_map = d_map
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_text = self.q_map.get(str(row["query_id"]), "")
        d_text = self.d_map.get(str(row["doc_id"]), "")
        label = float(row["0"])  # when RAG is correct, label is 1.0

        inputs = self.tokenizer(
            q_text,
            d_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


def train_one_fold(train_ds, device, model_name, args):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(ids, attention_mask=mask)
            loss = loss_fn(outputs.logits.view(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
    return model


def predict_top5(model, tokenizer, device, target_jsonl_data, query_map):
    model.eval()
    results = []

    for data in tqdm(target_jsonl_data, desc="Predicting Top-5"):
        q_id = data["question_id"]
        query_text = query_map.get(q_id, query_map.get(f"triviaqa::olmes:{q_id}", ""))

        top5_docs = [ctx["retrieval text"] for ctx in data["ctxs"][:5]]
        pairs = [[query_text, d_text] for d_text in top5_docs]

        inputs = tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits.view(-1)).cpu().numpy()

        res = {"query_id": q_id}
        for i, p in enumerate(probs):
            res[f"ce_score_{i+1}"] = p
        results.append(res)
    return results


def main(args):
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    query_map = json2dict(f"data/{args.data_type}/query_map.json")
    doc_map = json2dict(f"data/{args.data_type}/doc_map.json")

    rag_df = pd.read_csv(f"data/{args.data_type}/train_{args.task_type}.csv")
    with open(args.jsonl_path, "r") as f:
        all_jsonl = [json.loads(line) for line in f]

    all_q_ids = rag_df["query_id"].unique()
    kf = KFold(n_splits=2, shuffle=True, random_state=args.seed)

    final_stacking_scores = []

    # 2-Fold Stacking Loop
    # Use 2-fold validation to avoid label leakage here;
    # CE scores for each query are produced by a model trained on the other set of queries.
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_q_ids)):
        print(f"Fold {fold+1}/2 Processing...")
        train_ids = all_q_ids[train_idx]
        val_ids = set(all_q_ids[val_idx])

        train_subset_df = rag_df[rag_df["query_id"].isin(train_ids)].reset_index(
            drop=True
        )
        train_ds = CEFoldDataset(train_subset_df, query_map, doc_map, tokenizer)
        print(f"Training on {len(train_ids)} queries...")
        model = train_one_fold(train_ds, device, model_name, args)

        print(f"Predicting on remaining {len(val_ids)} queries...")
        val_jsonl = [
            item
            for item in all_jsonl
            if item["question_id"] in val_ids
            or f"triviaqa::olmes:{item['question_id']}" in val_ids
        ]
        fold_scores = predict_top5(model, tokenizer, device, val_jsonl, query_map)
        final_stacking_scores.extend(fold_scores)

    pd.DataFrame(final_stacking_scores).to_csv(
        f"data/{args.data_type}/ce_score_top5_train_stacking.csv", index=False
    )

    # Optional: Train on full data for test prediction
    # full_train_ds = CEFoldDataset(rag_df, query_map, doc_map, tokenizer)
    # final_model = train_one_fold(full_train_ds, device, model_name, args)


def setup_seed(seed):
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--data_type", type=str, default="triviaqa_20000")
    parser.add_argument("--task_type", type=str, default="local")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
