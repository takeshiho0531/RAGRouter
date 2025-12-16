import os
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import argparse
from src.utils import json2dict # User's existing utility

# --- 1. Feature Engineering ---
import math
from collections import Counter

class FeatureExtractor:
    def __init__(self, train_queries=None, train_docs=None, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        
        # === 追加: 単語の重要度(IDF)を計算して辞書にしておく ===
        # これにより "is" や "the" などの無駄な一致を無視し、
        # "Qwen" や "Router" のような重要単語の一致を高く評価できます
        self.word_idf = {}
        if train_queries is not None and train_docs is not None:
            self._compute_idf(train_queries + train_docs)
            
    def _compute_idf(self, texts):
        print("Computing IDF for lexical features...")
        doc_count = len(texts)
        word_counter = Counter()
        for text in texts:
            # 簡易トークナイズ (日本語ならここをMeCab等に変えるとさらに精度UP)
            words = set(text.lower().split())
            word_counter.update(words)
            
        for word, freq in word_counter.items():
            # 一般的な IDF の式: log(N / (n + 1))
            self.word_idf[word] = math.log(doc_count / (freq + 1))
            
    def _get_lexical_features(self, queries, docs):
        features = []
        for q, d in zip(queries, docs):
            q_tokens = set(q.lower().split())
            d_tokens = set(d.lower().split())
            
            if len(q_tokens) == 0:
                features.append([0.0, 0.0, 0.0, 0.0]) # 次元数を合わせる
                continue

            intersection = q_tokens.intersection(d_tokens)
            
            # 1. 既存: 単純な一致率 (Jaccard)
            union = q_tokens.union(d_tokens)
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0
            
            # 2. 既存: 長さ比率
            len_ratio = len(d) / len(q) if len(q) > 0 else 0
            
            # === 3. 新規: IDF重み付き一致スコア (これがCross-Encoderの代わりになる！) ===
            # 重要単語が一致していればスコアが跳ね上がる
            idf_overlap_score = sum(self.word_idf.get(w, 0) for w in intersection)
            
            # === 4. 新規: クエリの重要単語カバー率 ===
            # クエリに含まれる重要単語(IDF高)のうち、何割がドキュメントにあるか？
            q_total_idf = sum(self.word_idf.get(w, 0) for w in q_tokens)
            weighted_overlap_ratio = idf_overlap_score / q_total_idf if q_total_idf > 0 else 0

            features.append([jaccard, len_ratio, idf_overlap_score, weighted_overlap_ratio])
            
        return np.array(features)

    def get_features(self, queries, docs):
        # 1. Embedding & Similarity (既存)
        q_embs = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        d_embs = self.model.encode(docs, convert_to_tensor=True, show_progress_bar=False)
        scores = torch.cosine_similarity(q_embs, d_embs).cpu().numpy()
        q_embs_np = q_embs.cpu().numpy()
        
        # 2. Lexical Features (新規追加)
        lexical_feats = self._get_lexical_features(queries, docs)
        
        # [Similarity(1) + Lexical(3) + QueryEmb(384)]
        # これでドキュメント側の情報が「1個」から「4個」に増えます
        features = np.hstack([
            scores.reshape(-1, 1), 
            lexical_feats, 
            q_embs_np
        ])
        return features

def prepare_data(args, extractor):
    """
    CSVとMapからデータを読み込み、特徴量Xとラベルyを作成する
    """
    query_map = json2dict(f'data/{args.data_type}/query_map.json')
    doc_map = json2dict(f'data/{args.data_type}/doc_map.json')
    
    datasets = {}
    for split in ['train', 'test']:
        rag_df = pd.read_csv(f'data/{args.data_type}/{split}_{args.task_type}.csv')
        norag_df = pd.read_csv(f'data/{args.data_type}/{split}.csv')
        
        # リスト作成
        queries = [query_map.get(str(qid), "") for qid in rag_df['query_id']]
        docs = [doc_map.get(str(did), "") for did in rag_df['doc_id']]
        
        # ラベル取得
        l_rag = rag_df['0'].values.astype(float)    # col name might vary
        l_norag = norag_df['1'].values.astype(float) # col name might vary
        
        # 特徴量抽出
        print(f"Extracting features for {split}...")
        X = extractor.get_features(queries, docs)
        
        datasets[split] = {
            'X': X,
            'l_rag': l_rag,
            'l_norag': l_norag
        }
    return datasets

# --- 2. LightGBM Implementation ---
def train_lightgbm(datasets):
    print("\n=== Training LightGBM Router ===")
    
    X_train = datasets['train']['X']
    l_rag_train = datasets['train']['l_rag']
    l_norag_train = datasets['train']['l_norag']
    
    mask = (l_rag_train != l_norag_train)
    X_train_filtered = X_train[mask]
    y_train_filtered = (l_rag_train[mask] == 1.0).astype(int)
    
    print(f"Training samples (conflict cases only): {len(X_train_filtered)}")
    
    train_data = lgb.Dataset(X_train_filtered, label=y_train_filtered)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        
        # === 変更点1: Extra Treesは維持（これが汎化の鍵） ===
        'extra_trees': True,
        
        # === 変更点2: モデルサイズは「小」に戻す ===
        # 63や100は過学習の元でした。31で十分表現できています。
        'num_leaves': 100,
        
        # === 変更点3: ブレーキの強さを「最適化」 ===
        # 10.0だと強すぎ、1.0だと弱すぎたので、5.0にします。
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        
        # === 変更点4: 「超低速学習」モード ===
        # ここが勝負所です。ゆっくり学習させることで、局所解（過学習）を避けます。
        # 'learning_rate': 0.005,
        
        # 過学習対策のダメ押し（葉っぱ1つに最低100個のデータを要求）
        # 'min_data_in_leaf': 100, 
        
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,
        'verbose': -1
    }

    # 学習率は0.01なので、回数は2000回くらいで十分収束します
    gbm = lgb.train(params, train_data, num_boost_round=5000)
    
    # 見つかった閾値を渡して評価
    print("\n=== [TRAIN] Data Evaluation ===")
    evaluate_router(gbm, datasets['train'], model_type='lightgbm')
    evaluate_router(gbm, datasets['test'], model_type='lightgbm')
    return gbm

# --- 3. MLP (PyTorch) Implementation ---
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1) # Logits
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(datasets, device='cuda', epochs=10):
    print("\n=== Training MLP Router ===")
    
    X_train = datasets['train']['X']
    l_rag_train = datasets['train']['l_rag']
    l_norag_train = datasets['train']['l_norag']
    
    # LightGBM同様、差異があるデータのみで学習
    mask = (l_rag_train != l_norag_train)
    X_train_filt = torch.tensor(X_train[mask], dtype=torch.float32).to(device)
    y_train_filt = torch.tensor((l_rag_train[mask] == 1.0).astype(float), dtype=torch.float32).unsqueeze(1).to(device)
    
    model = SimpleMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_filt)
        loss = loss_fn(logits, y_train_filt)
        loss.backward()
        optimizer.step()
        
        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1}, Loss: {loss.item():.4f}")
            
    evaluate_router(model, datasets['test'], model_type='mlp', device=device)
    return model

# --- 4. Unified Evaluation Logic ---
def evaluate_router(model, test_data, model_type='lightgbm', device='cuda', threshold=0.5):
    X_test = test_data['X']
    l_rag = test_data['l_rag']
    l_norag = test_data['l_norag']
    
    # Predict
    if model_type == 'lightgbm':
        probs = model.predict(X_test)
        # ここで最適化された閾値を使う
        preds_is_rag = (probs > threshold) 
    else: # mlp
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = model(inputs)
            # MLPの場合も厳密にはlogitの閾値を調整すべきですが、一旦0基準で
            preds_is_rag = (logits > 0).cpu().numpy().flatten()
            
    total_samples = len(l_rag)
    router_optimal_choice = 0
    
    for i in range(total_samples):
        r_correct = (l_rag[i] == 1.0)
        n_correct = (l_norag[i] == 1.0)
        chose_rag = preds_is_rag[i]

        if r_correct == n_correct:
            router_optimal_choice += 1
        else:
            if r_correct and chose_rag:
                router_optimal_choice += 1
            elif n_correct and not chose_rag:
                router_optimal_choice += 1
                
    optimal_acc = router_optimal_choice / total_samples
    
    print(f"[{model_type.upper()}] Results (Threshold: {threshold:.3f}):")
    print(f"  Router Acc (Optimal Choice): {optimal_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='webq')
    parser.add_argument('--task_type', type=str, default='local')
    args = parser.parse_args()
    
    # 1. 特徴量抽出 (Embedding等)
    # 実際の運用ではencode済みのpickle等をロードする方が高速です
    # extractor = FeatureExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
    # datasets = prepare_data(args, extractor)
    # mainブロック内

    # データをロード (Map読み込み)
    query_map = json2dict(f'data/{args.data_type}/query_map.json')
    doc_map = json2dict(f'data/{args.data_type}/doc_map.json')

    # 全テキストをリスト化して渡す（IDF計算用）
    all_texts = list(query_map.values()) + list(doc_map.values())

    extractor = FeatureExtractor(
        train_queries=list(query_map.values()), # IDF計算用に渡す
        train_docs=list(doc_map.values()),      # IDF計算用に渡す
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    datasets = prepare_data(args, extractor)
    
    # 2. LightGBMで学習・評価
    train_lightgbm(datasets)
    
    # 3. MLPで学習・評価
    # train_mlp(datasets)