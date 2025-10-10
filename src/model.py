import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEncoder(nn.Module):
    def __init__(self, encoder):
        super(CrossEncoder, self).__init__()
        self.encoder = encoder
        
    def forward(self, text):
        embedding = self.encoder.bert(**text)
        embedding = embedding.pooler_output
        return embedding


class TextEncoder(nn.Module):
    def __init__(self, encoder):
        super(TextEncoder, self).__init__()
        self.encoder = encoder
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, text):
        embedding = self.encoder(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        embedding = self.mean_pooling(embedding, text['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class RAGRouter(nn.Module):
    def __init__(self, num_models=15, model_embedding_dim=768, text_dim=768, cross_dim=384, similarity_function='cos', text_encoder=None, query_encoder=None, doc_encoder=None):
        super(RAGRouter, self).__init__()
        self.k_encoder = nn.Embedding(num_models, model_embedding_dim)
        self.i_encoder = nn.Embedding(num_models, model_embedding_dim)
        std_dev = 0.78
        with torch.no_grad():
            nn.init.normal_(self.k_encoder.weight, mean=0, std=std_dev)
            nn.init.normal_(self.i_encoder.weight, mean=0, std=std_dev)
        
        p_list = ['encoder.layer.11', 'encoder.layer.10', 'bert.encoder.layer.11', 'bert.encoder.layer.10']
        for name, param in query_encoder.named_parameters():
            if p_list[0] in name or p_list[1] in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in text_encoder.named_parameters():
            if p_list[2] in name or p_list[3] in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.query_encoder = TextEncoder(query_encoder)
        self.doc_encoder = TextEncoder(doc_encoder)
        self.text_encoder = CrossEncoder(text_encoder)

        self.query_proj = nn.Linear(text_dim, model_embedding_dim)
        self.text_proj = nn.Linear(text_dim + cross_dim, model_embedding_dim)
        self.attn_fusion = nn.MultiheadAttention(
            embed_dim=model_embedding_dim,
            num_heads=8,
            batch_first=True
        )

        self.similarity_function = similarity_function

    def compute_similarity(self, input1, input2):
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (torch.norm(input1,dim=1).unsqueeze(1) * torch.norm(input2,dim=1).unsqueeze(0))
        else:
            return input1 @ input2.T
    
    def forward(self, text, query, doc):
        query_embedding = self.query_encoder(query)
        query_embedding = self.query_proj(query_embedding)
        x0 = self.compute_similarity(query_embedding, self.k_encoder.weight)

        text_embedding = self.text_encoder(text)
        doc_embedding = self.doc_encoder(doc)
        text_embedding = self.text_proj(torch.cat([doc_embedding, text_embedding], dim=1))
        fusion_embedding, _ = self.attn_fusion(
            self.i_encoder.weight,
            text_embedding,
            text_embedding
        )
        x1 = self.compute_similarity(query_embedding, self.k_encoder.weight + fusion_embedding)
        return x1, x0
