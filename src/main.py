import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import uuid
import torch
import random
import warnings
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pprint import pprint
from torch.optim import AdamW, lr_scheduler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from utils import *
from dataset import get_dataloader
from model import RAGRouter
import torch.nn.functional as F
warnings.simplefilter(action='ignore', category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def contrastive_loss(x1, x0, labels, labelso, temperature=0.2):
    batch_size = x1.size(0)
    total_loss = 0.0
    num_pos = 0
    
    for i in range(batch_size):
        pos_x1_mask = labels[i].bool()
        pos_x1_scores = x1[i][pos_x1_mask]

        pos_x0_mask = labelso[i].bool()
        pos_x0_scores = x0[i][pos_x0_mask]

        neg_x1_mask = ~pos_x1_mask
        neg_x1_scores = x1[i][neg_x1_mask]

        neg_x0_mask = ~pos_x0_mask
        neg_x0_scores = x0[i][neg_x0_mask]
        
        pos_scores = torch.cat([pos_x1_scores, pos_x0_scores])
        neg_scores = torch.cat([neg_x1_scores, neg_x0_scores])
        for pos_score in pos_scores:
            if neg_scores.numel() == 0:
                continue
            numerator = torch.exp(pos_score / temperature)
            denominator = numerator + torch.sum(torch.exp(neg_scores / temperature))
            loss = -torch.log(numerator / denominator)
            total_loss += loss
            num_pos += 1

    if num_pos == 0:
        return torch.tensor(0.0, device=x1.device)
    
    return total_loss / num_pos

def binary_success_loss(cls_preds, labels):
    # labels[:, 0]: label for RAG success
    # cls_preds[:, 0]: logit for RAG success
    rag_logit = cls_preds[:, 0]
    rag_label = labels[:, 0]

    loss = F.binary_cross_entropy_with_logits(rag_logit, rag_label)
    return loss

def evaluate(net, test_loader, device, epoch):
    net.eval()
    total_loss = 0.0
    correct = 0
    num_samples = 0

    with torch.no_grad():
        for texts, querys, docs, labels, labelso in tqdm(test_loader):
            texts, querys, docs, labels, labelso = texts.to(device), querys.to(device), docs.to(device), labels.to(device), labelso.to(device)
            cls_preds, _ = net(texts, querys, docs) # cls_predsoは使わない

            loss = binary_success_loss(cls_preds, labels)
            total_loss += loss.item() * labels.size(0)

            rag_pred = (cls_preds[:, 0] > 0).float()
            rag_true = labels[:, 0]

            correct += (rag_pred == rag_true).sum().item()
            num_samples += labels.size(0)

    mean_loss = total_loss / num_samples
    accuracy = correct / num_samples

    print(f"Test Loss: {mean_loss:.6f}, Success Predictor Acc: {accuracy:.4f}")
    net.train()
    return mean_loss, accuracy

# Main training loop
def train(net, train_loader, test_loader, device): # check evaluate() for reference
    optimizer = AdamW(net.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        net.train()
        total_loss = 0.0
        total_used = 0 # number of samples used for loss calculation
        correct = 0
        num_samples = 0

        for texts, querys, docs, labels, labelso in tqdm(train_loader):
            texts, querys, docs, labels, labelso = texts.to(device), querys.to(device), docs.to(device), labels.to(device), labelso.to(device)
            optimizer.zero_grad()
            cls_preds, _ = net(texts, querys, docs)
            loss = binary_success_loss(cls_preds, labels)

            loss.backward()
            optimizer.step()

            batch_size_current = labels.size(0)
            total_loss += loss.item() * labels.size(0)

            rag_pred = (cls_preds[:, 0] > 0).float()
            correct += (rag_pred == labels[:, 0]).sum().item()
            total_used += batch_size_current
            num_samples += labels.size(0)

        mean_loss = (total_loss / total_used) if total_used > 0 else 0.0
        train_acc = (correct / total_used) if total_used > 0 else 0.0
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Train (binary) Loss: {mean_loss:.6f}, Acc: {train_acc:.4f}, Used: {total_used}/{num_samples}")

        test_loss, router_acc = evaluate(net, test_loader, device, epoch)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Test (binary) Loss: {test_loss:.6f}, Acc: {router_acc:.4f}")

        torch.save(net, f'checkpoints/{args.id}/{args.id}-{epoch + 1}.pth')
        scheduler.step()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--num_models", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--id', type=str, default=str(uuid.uuid1().hex[:12]))
    parser.add_argument('--data_type', type=str, default='webq')
    parser.add_argument('--task_type', type=str, default='local')
    parser.add_argument('--lambda1', type=float, default=2.0)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.95)
    args = parser.parse_args()
    pprint(vars(args))
    setup_seed(args.seed)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('results'):
        os.makedirs('results')
    os.makedirs(f'results/{args.id}')
    os.makedirs(f'checkpoints/{args.id}')
    cross_encoder_path = "cross-encoder/ms-marco-MiniLM-L12-v2"
    encoder_path = "sentence-transformers/all-mpnet-base-v2"


    print("Loading dataset...")
    cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_path)
    tokenizer_all = AutoTokenizer.from_pretrained(cross_encoder_path)
    encoder_model = AutoModel.from_pretrained(encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    train_loader, test_loader = get_dataloader(
        tokenizer=tokenizer,
        tokenizer_all=tokenizer_all,
        batch_size=args.batch_size,
        data_type=args.data_type,
        task_type=args.task_type,
        num_models=args.num_models,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Initializing model...")
    net = RAGRouter(
        num_models=args.num_models,
        model_embedding_dim=args.embedding_dim,
        text_encoder=cross_encoder,
        query_encoder=encoder_model,
        doc_encoder=encoder_model,
    )
    net.to(device)
    print("Training model...")
    train(net=net, train_loader=train_loader, test_loader=test_loader, device=device)
