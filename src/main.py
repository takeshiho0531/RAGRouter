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

def binary_route_loss(cls_preds, cls_predso, labels, labelso):
    # A: 0.6B+RAG (x1[:,0]), B: 14B noRAG (x0[:,1])
    a_logit = cls_preds[:, 0]
    b_logit = cls_predso[:, 1]
    diff = a_logit - b_logit # probability difference

    a = labels[:, 0]
    b = labelso[:, 1]
    mask = (a + b == 1) # calculate loss only when there is a difference: A is correct and B is not, or vice versa

    if mask.sum() == 0:
        return torch.tensor(0.0, device=cls_preds.device), mask

    y = a[mask]  # ground truth
    loss = F.binary_cross_entropy_with_logits(diff[mask], y)
    return loss, mask

def evaluate(net, test_loader, device, epoch):
    net.eval()
    total_loss = 0.0
    total_used = 0  # number of samples used for loss calculation. different samples only
    correct_all = 0 # for calculating router accuracy over all samples
    num_samples = 0

    with torch.no_grad():
        for texts, querys, docs, labels, labelso in tqdm(test_loader):
            texts, querys, docs, labels, labelso = texts.to(device), querys.to(device), docs.to(device), labels.to(device), labelso.to(device)
            cls_preds, cls_predso = net(texts, querys, docs)

            # calculate loss only when there is a difference
            loss, mask = binary_route_loss(cls_preds, cls_predso, labels, labelso)
            used = int(mask.sum().item())
            total_loss += loss.item() * used
            total_used += used

            # accuracy output: use all query data
            batch_size = labels.shape[0]
            num_samples += batch_size

            diff = cls_preds[:, 0] - cls_predso[:, 1]
            pred_is_A = (diff > 0)

            # ground truth
            is_A_correct = labels[:, 0].bool()
            is_B_correct = labelso[:, 1].bool()

            # When router chooses A: 0.6B+RAG is better, which is correct
            case_A_better = (is_A_correct & ~is_B_correct) & pred_is_A
            
            # When router chooses A: 14B is better, which is correct
            case_B_better = (~is_A_correct & is_B_correct) & (~pred_is_A)
            
            # When both are correct or both are incorrect -> whichever router chooses is correct
            case_tie = (is_A_correct == is_B_correct)

            batch_correct = (case_A_better | case_B_better | case_tie).sum().item()
            correct_all += batch_correct

    mean_loss = (total_loss / total_used) if total_used > 0 else 0.0
    
    # calculate router accuracy over all samples (not just used samples)
    router_acc = correct_all / num_samples if num_samples > 0 else 0.0

    print(f"Test (binary) Loss: {mean_loss:.6f}, Oracle Match Acc: {router_acc:.4f}, Total Samples: {num_samples}")

    net.train()
    return mean_loss, router_acc


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
            cls_preds, cls_predso = net(texts, querys, docs)
            loss, mask = binary_route_loss(cls_preds, cls_predso, labels, labelso)
            used = int(mask.sum().item())
            if used == 0:
                num_samples += labels.shape[0]
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * used
            total_used += used
            num_samples += labels.shape[0]

            diff = cls_preds[:, 0] - cls_predso[:, 1]
            y = labels[:, 0][mask]
            pred = (diff[mask] > 0).float()
            correct += int((pred == y).sum().item())

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
