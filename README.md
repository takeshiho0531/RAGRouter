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
