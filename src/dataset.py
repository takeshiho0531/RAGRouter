import torch
from torch.utils.data import Dataset, DataLoader
from utils import *


class RouteDataset(Dataset):
    def __init__(
            self, 
            file_path, 
            file_patho,
            query_map: pd.DataFrame,
            doc_map: pd.DataFrame=None,
            tokenizer_all=None,
            tokenizer=None,
            target_max_token_len=512,
            ) -> None:
        super().__init__()
        self.data = read_csv(file_path)
        self.datao = read_csv(file_patho)
        self.query_map = query_map
        self.doc_map = doc_map
        self.tokenizer_all = tokenizer_all
        self.tokenizer = tokenizer
        self.target_max_token_len = target_max_token_len
    
    def get_text_feature_all(self, query, doc):
        text_id = self.tokenizer_all(
            [query], [doc],  
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_id['input_ids'] = text_id.input_ids.squeeze(-2)
        text_id['token_type_ids'] = text_id.token_type_ids.squeeze(-2)
        text_id['attention_mask'] = text_id.attention_mask.squeeze(-2)
        return text_id
    
    def get_text_feature(self, text):
        text_id = self.tokenizer(
            text,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_id['input_ids'] = text_id.input_ids.squeeze(-2)
        text_id['attention_mask'] = text_id.attention_mask.squeeze(-2)
        return text_id

    def __getitem__(self, index):
        query = self.get_text_feature(self.query_map[str(self.data[index]['query_id'])])
        doc = self.get_text_feature(self.doc_map[str(self.data[index]['doc_id'])])
        text = self.get_text_feature_all(self.query_map[str(self.data[index]['query_id'])], self.doc_map[str(self.data[index]['doc_id'])])
        label = torch.tensor([
            self.data[index]['0'],
            self.data[index]['1'],
            self.data[index]['2'],
            self.data[index]['3'],
            self.data[index]['4'],
            self.data[index]['5'],
            self.data[index]['6'],
            self.data[index]['7'],
            self.data[index]['8'],
            self.data[index]['9'],
            self.data[index]['10'],
            self.data[index]['11'],
            self.data[index]['12'],
            self.data[index]['13'],
            self.data[index]['14'],
        ], dtype=torch.float32)
        labelo = torch.tensor([
            self.datao[index]['0'],
            self.datao[index]['1'],
            self.datao[index]['2'],
            self.datao[index]['3'],
            self.datao[index]['4'],
            self.datao[index]['5'],
            self.datao[index]['6'],
            self.datao[index]['7'],
            self.datao[index]['8'],
            self.datao[index]['9'],
            self.datao[index]['10'],
            self.datao[index]['11'],
            self.datao[index]['12'],
            self.datao[index]['13'],
            self.datao[index]['14'],
        ], dtype=torch.float32)
        return text, query, doc, label, labelo

    def __len__(self):
        return len(self.data)


def get_dataloader(tokenizer, tokenizer_all, batch_size, data_type=None, task_type=None):
    train_dataset = RouteDataset(
        file_path=f'data/{data_type}/train_{task_type}.csv',
        file_patho=f'data/{data_type}/train.csv',
        query_map=json2dict(f'data/{data_type}/query_map.json'),
        doc_map=json2dict(f'data/{data_type}/doc_map.json'),
        tokenizer_all=tokenizer_all,
        tokenizer=tokenizer,
    )
    test_dataset = RouteDataset(
        file_path=f'data/{data_type}/test_{task_type}.csv',
        file_patho=f'data/{data_type}/test.csv',
        query_map=json2dict(f'data/{data_type}/query_map.json'),
        doc_map=json2dict(f'data/{data_type}/doc_map.json'),
        tokenizer_all=tokenizer_all,
        tokenizer=tokenizer,
    )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
