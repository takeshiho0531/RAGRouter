import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from utils import json2dict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SimpleRouteDataset(Dataset):
    def __init__(
            self, 
            file_path_rag,      # train_local.csv (0.6B+RAG results)
            file_path_norag,    # train.csv (14B inference results)
            query_map, 
            doc_map, 
            tokenizer, 
            max_len=512,
            rag_col_idx='0',    # column name in CSV where RAG model correctness is stored
            norag_col_idx='1'   # column name in CSV where No-RAG model correctness is stored
            ):
        self.data_rag = pd.read_csv(file_path_rag)
        self.data_norag = pd.read_csv(file_path_norag)
        self.query_map = query_map
        self.doc_map = doc_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rag_col = rag_col_idx
        self.norag_col = norag_col_idx

    def __len__(self):
        return len(self.data_rag)

    def __getitem__(self, index):
        q_id = str(self.data_rag.iloc[index]['query_id'])
        d_id = str(self.data_rag.iloc[index]['doc_id'])

        query_text = self.query_map.get(q_id, "")
        doc_text = self.doc_map.get(d_id, "")

        # cross-encoder style input
        inputs = self.tokenizer(
            query_text,
            doc_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # labels: 1.0 if correct, 0.0 if incorrect
        label_rag = float(self.data_rag.iloc[index][self.rag_col])
        label_norag = float(self.data_norag.iloc[index][self.norag_col])

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label_rag': torch.tensor(label_rag, dtype=torch.float),
            'label_norag': torch.tensor(label_norag, dtype=torch.float)
        }

def evaluate(model, dataloader, device):
    model.eval()

    total_system_correct = 0 # the number of samples where the chosen model answered correctly to the query
    total_router_choice_correct = 0 # the number of samples where the correct model was chosen by the router
    
    total_samples = 0
    total_loss = 0
    total_diff = 0
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_rag = batch['label_rag'].to(device)
            label_norag = batch['label_norag'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1)
            
            # logit > 0: choose RAG model, predicts_is_rag True
            # logit <=0: choose No-RAG model, predicts_is_rag False
            preds_is_rag = (logits > 0) 

            for i in range(len(label_rag)):
                l_rag = label_rag[i].item()
                l_norag = label_norag[i].item()
                chose_rag = preds_is_rag[i].item()

                if chose_rag:
                    if l_rag == 1.0:
                        total_system_correct += 1
                else:
                    if l_norag == 1.0:
                        total_system_correct += 1

                if l_rag == l_norag: # both correct or both incorrect
                    total_router_choice_correct += 1
                else:
                    if l_rag == 1.0 and chose_rag: # RAG is correct and RAG was chosen
                        total_router_choice_correct += 1

                    elif l_norag == 1.0 and not chose_rag: # No-RAG is correct and No-RAG was chosen
                        total_router_choice_correct += 1

                if l_rag != l_norag:
                    target = 1.0 if l_rag == 1.0 else 0.0
                    total_loss += loss_fn(logits[i], torch.tensor(target, device=device)).item()
                    total_diff += 1 

                total_samples += 1

    avg_system_acc = total_system_correct / total_samples if total_samples > 0 else 0
    avg_choice_acc = total_router_choice_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_diff if total_diff > 0 else 0
    
    return avg_system_acc, avg_choice_acc, avg_loss


def train(args):
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # num_labels=1; logit positive: 0.6B+rag, negative: no-rag 14B
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    print("Loading data...")
    query_map = json2dict(f'data/{args.data_type}/query_map.json')
    doc_map = json2dict(f'data/{args.data_type}/doc_map.json')

    train_ds = SimpleRouteDataset(
        file_path_rag=f'data/{args.data_type}/train_{args.task_type}.csv',
        file_path_norag=f'data/{args.data_type}/train.csv',
        query_map=query_map, doc_map=doc_map, tokenizer=tokenizer
    )
    test_ds = SimpleRouteDataset(
        file_path_rag=f'data/{args.data_type}/test_{args.task_type}.csv',
        file_path_norag=f'data/{args.data_type}/test.csv',
        query_map=query_map, doc_map=doc_map, tokenizer=tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = 0.0

    print("Start Training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_effective = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_rag = batch['label_rag'].to(device)
            label_norag = batch['label_norag'].to(device)

            # use as training data only when there is a difference; rag is correct and no-rag is not, or vice versa
            mask = (label_rag + label_norag) == 1
            if mask.sum() == 0:
                continue

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1)

            active_logits = logits[mask]
            active_targets = label_rag[mask]
            # Since only one of them (0.6B+RAG vs non-RAG 14B) is correct, it's possible to use label_rag as the target directly
            loss = loss_fn(active_logits, active_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_effective += mask.sum().item()

            preds = (active_logits > 0).float()
            correct += (preds == active_targets).sum().item()

        train_loss = total_loss / total_effective if total_effective > 0 else 0
        train_acc = correct / total_effective if total_effective > 0 else 0

        sys_acc, choice_acc, eval_loss = evaluate(model, test_loader, device)
        print(f"  Eval Loss (diff-only): {eval_loss:.4f}")
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  System Acc (End-to-End): {sys_acc:.4f} (the accuracy of data answered correctly by the chosen model)")
        print(f"  Router Acc (Choice Only): {choice_acc:.4f} (the accuracy of the router choosing the correct model)")

        if sys_acc > best_acc: 
            best_acc = sys_acc
            save_path = f'checkpoints/{args.id}_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  Saved Best Model based on System Acc")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--id', type=str, default='simple_router')
    parser.add_argument('--data_type', type=str, default='webq')
    parser.add_argument('--task_type', type=str, default='local')
    
    args = parser.parse_args()
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    train(args)