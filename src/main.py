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


def evaluate(net, test_loader, device, epoch):
    net.eval()
    loss_fn_cls = nn.BCEWithLogitsLoss()
    total_loss = 0
    loss_cls = 0
    loss_ct = 0
    num_samples = 0
    res = {'pred': [], 'label': []}
    with torch.no_grad():
        for texts, querys, docs, labels, labelso in tqdm(test_loader):
            texts, querys, docs, labels, labelso = texts.to(device), querys.to(device), docs.to(device), labels.to(device), labelso.to(device)
            cls_preds, cls_predso = net(texts, querys, docs)

            loss = args.lambda1 * loss_fn_cls(cls_preds, labels) + args.lambda1 * loss_fn_cls(cls_predso, labelso) + contrastive_loss(cls_preds, cls_predso, labels, labelso, args.tau)
            total_loss += loss.item()
            loss_cls += loss_fn_cls(cls_preds, labels).item()
            loss_ct += contrastive_loss(cls_preds, cls_predso, labels, labelso, args.tau)
            num_samples += labels.shape[0]

            res['pred'].extend(torch.sigmoid(cls_preds).flatten().tolist())
            res['label'].extend(labels.flatten().tolist())

    save_csv(res, f'results/{args.id}/res-{args.id}-{epoch + 1}.csv')
    router_acc = eval_router(read_csv(f'results/{args.id}/res-{args.id}-{epoch + 1}.csv'), all=num_samples, num_models=args.num_models)
    print(f'Test Loss: {loss_cls / (num_samples * args.num_models)}, {loss_ct}')
    mean_loss = total_loss / (num_samples * args.num_models)
    net.train()
    return mean_loss, router_acc


# Main training loop
def train(net, train_loader, test_loader, device):
    optimizer = AdamW(net.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_fn_cls = nn.BCEWithLogitsLoss()

    for epoch in range(args.num_epochs):
        net.train()
        total_loss = 0
        loss_cls = 0
        loss_ct = 0
        for texts, querys, docs, labels, labelso in tqdm(train_loader):
            texts, querys, docs, labels, labelso = texts.to(device), querys.to(device), docs.to(device), labels.to(device), labelso.to(device)
            optimizer.zero_grad()
            cls_preds, cls_predso = net(texts, querys, docs)
            loss = args.lambda1 * loss_fn_cls(cls_preds, labels) + args.lambda1 * loss_fn_cls(cls_predso, labelso) + contrastive_loss(cls_preds, cls_predso, labels, labelso, args.tau)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_cls += loss_fn_cls(cls_preds, labels).item()
            loss_ct += contrastive_loss(cls_preds, cls_predso, labels, labelso, args.tau)

        train_loss = total_loss / len(train_loader)
        loss_cls = loss_cls / len(train_loader)
        loss_ct = loss_ct / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss}, {loss_cls}, {loss_ct}")
        test_loss, router_acc = evaluate(net, test_loader, device, epoch)
        print(f"Test Loss: {test_loss}, Accuracy: {router_acc}")
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
        task_type=args.task_type
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
