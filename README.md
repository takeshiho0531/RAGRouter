# Router Preformance
- this commit, src/lightgbm_router.py, triviaqa_full, trust_14b 0.85
```
Optimizing Delta (MCC - Penalty)...
energy_penalty_weight: 0.15
Aggressive Delta Optimized: 0.375
=== System Performance Evaluation ===
Total System Accuracy: 0.6366 (Baseline 14B: 0.6132)
RAG Call Rate: 15.02%
=== Router Decision Quality (Is RAG really needed?) ===
Router Selection Accuracy: 0.8033
├─ Precision (batting accuracy / effective RAG): 0.2733
├─ Recall    (coverage / rescue rate): 0.3191
└─ MCC       (overall plate discipline): 0.1816
Confusion Matrix (Selection Strategy):
                Pred: 14B | Pred: RAG
Actual: 14B-OK:     5831 |       835 (Overkill or Waste)
Actual: NeedRAG:     670 |       314 (Saved!)
--- Feature Importance (14B) Top 10 ---
           f             v
10   q_pca_3  47604.561095
7    q_pca_0  36507.856471
11   q_pca_4  16306.642071
9    q_pca_2  15241.300001
20  q_pca_13  15225.657823
8    q_pca_1  13259.520516
13   q_pca_6  10759.955159
14   q_pca_7  10563.693863
24  q_pca_17  10329.205142
16   q_pca_9  10133.222225
--- Feature Importance (0.6B+RAG) Top 10 ---
            f              v
1      ce_max  362982.327156
39    d_pca_0   31224.583567
0     ce_top1   23666.918295
15    q_pca_8   18405.996265
6   agreement   13777.885805
43    d_pca_4   10769.355595
7     q_pca_0   10718.352192
4       q_len   10634.244312
42    d_pca_3    9729.045513
40    d_pca_1    8976.483500
```
<br>

- this commit, src/direct_predictor.py, triviaqa_full
```
=== System Performance Evaluation ===
Total System Accuracy: 0.6260 (Baseline 14B: 0.6132)
RAG Call Rate: 3.82%
=== Router Decision Quality (NeedRAG detection) ===
Router Selection Accuracy: 0.8654
├─ Precision (effective RAG): 0.4212
├─ Recall    (rescue rate):   0.1250
└─ MCC       (overall):       0.1741
Confusion Matrix (NeedRAG target):
                Pred: noRAG | Pred: RAG
Actual: noNeed:    6497 |       169 (Overkill/Waste)
Actual: NeedRAG:    861 |       123 (Saved!)
--- Feature Importance (NeedRAG) Top 10 ---
          f              v
1    ce_max  126068.968639
39  d_pca_0   88678.006817
42  d_pca_3   54323.648632
7   q_pca_0   38333.225603
10  q_pca_3   33706.846739
40  d_pca_1   27758.218220
15  q_pca_8   27666.524347
45  d_pca_6   20573.294090
4     q_len   19402.841909
47  d_pca_8   16661.484509
```

## Prediction Performance of 14B's Success
this commit, src/14b_predictor.py, triviaqa_full
```
# pca_dim 16
========================================
  14B Success Prediction Results
========================================
Accuracy:  0.6536
ROC-AUC:   0.6683
MCC:       0.2232
Confusion Matrix:
                Pred: Fail | Pred: Success
Actual: Fail:          990 |         1969
Actual: Success:       681 |         4010


# pca_dim 32
========================================
  14B Success Prediction Results
========================================
Accuracy:  0.6561
ROC-AUC:   0.6768
MCC:       0.2321
Confusion Matrix:
                Pred: Fail | Pred: Success
Actual: Fail:         1053 |         1906
Actual: Success:       725 |         3966

# pca_dim 64
========================================
  14B Success Prediction Results
========================================
Accuracy:  0.6607
ROC-AUC:   0.6784
MCC:       0.2434
Confusion Matrix:
                Pred: Fail | Pred: Success
Actual: Fail:         1074 |         1885
Actual: Success:       711 |         3980

# pca_dim 128
========================================
  14B Success Prediction Results
========================================
Accuracy:  0.6613
ROC-AUC:   0.6846
MCC:       0.2448
Confusion Matrix:
                Pred: Fail | Pred: Success
Actual: Fail:         1072 |         1887
Actual: Success:       704 |         3987
```
<br>

# RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented Language Models

## Environment Setup

We recommend using Conda to manage your Python environment:

```bash
conda create -n ragrouter python=3.12
conda activate ragrouter
pip install -r requirements.txt
```

## Data Preparation

The preprocessed datasets are provided in the `data` folder, so you can start directly without running additional preprocessing steps.

- `model_map.json` – Mapping between model IDs (0~14) and model names.
- `doc_map.json` – Mapping between document IDs and their corresponding document contents.
- `query_map.json` – Mapping between query IDs and query texts.
- `train.csv` / `test.csv` – 0/1 correctness labels for model responses without RAG.
- `train_local.csv` / `test_local.csv` – 0/1 correctness labels for responses with local retrieval.
- `train_online.csv` / `test_online.csv` – 0/1 correctness labels for responses with online retrieval.

## Training and Evaluation

Run the following command to train and evaluate the RAGRouter:

```bash
python src/main.py --data_type <dataset_name> --task_type <retrieval_type>
```

### Key Arguments

- `--embedding_dim`: Dimension of the knowledge representation and RAG capability vectors (default: 768)  
- `--num_models`: Number of candidate models in the model pool (default: 15)  
- `--batch_size`: Training batch size (default: 64)  
- `--num_epochs`: Number of training epochs (default: 10)  
- `--learning_rate`: Learning rate (default: 5e-5)  
- `--data_type`: Name of the dataset to use  
- `--task_type`: Retrieval mode type: 'local' or 'online'  
- `--lambda1`: Coefficient for the classification loss term (default: 2)  
- `--tau`: Temperature coefficient for contrastive learning loss (default: 0.2)
