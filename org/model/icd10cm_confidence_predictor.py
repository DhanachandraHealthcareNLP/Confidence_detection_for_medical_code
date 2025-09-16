import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)  # learnable weights → scalar score per embedding

    def forward(self, embs):
        """
        embs: Tensor of shape (num_texts, embed_dim)
        Returns: (embed_dim,) pooled representation
        """
        # Compute attention scores
        scores = self.attn(embs).squeeze(-1)  # (num_texts,)
        weights = F.softmax(scores, dim=0)  # normalize to [0,1]

        # Weighted sum
        pooled = torch.sum(weights.unsqueeze(-1) * embs, dim=0)  # (embed_dim,)
        return pooled


class RobertaEmbedder(nn.Module):
    def __init__(self, model_name="roberta-base", device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size
        self.device = device
        self.to(device)

    def forward(self, texts, max_length=512):
        """
        texts: List[str] (one datapoint with multiple justifications/evidences)
        returns: (num_texts, embed_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**enc)  # (batch, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state

        # Mean pooling (ignoring padding)
        mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = torch.sum(last_hidden * mask, 1)
        lengths = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_hidden / lengths  # (batch, hidden_size)

        return mean_pooled


class ICDConfidenceModel(nn.Module):
    def __init__(self, embed_model, device, hidden_dim=512, dropout=0.2, config=None):
        super().__init__()
        self.device = device
        self.embed_model = embed_model
        self.config = config
        if self.config["use_roberta_for_emb"]:
            self.embed_model = RobertaEmbedder(self.config["roberta_model_name"], device=device)
        self.embed_dim = embed_model.get_sentence_embedding_dimension()
        self.attn_pool = AttentionPooling(self.embed_dim)
        input_dim = self.embed_dim * 6  # code + 5 MEAT fields
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_field(self, texts_list):
        batch_embs = []
        for texts in texts_list:
            if not texts:
                emb = torch.zeros(self.embed_dim, device=self.device)
            else:
                # Filter out empty strings
                valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]

                if len(valid_texts) == 0:
                    emb = torch.zeros(self.embed_dim, device=self.device)
                else:
                    embs = self.embed_model.encode(
                        valid_texts,
                        convert_to_tensor=True,
                        device=self.device
                    )  # (num_texts, embed_dim)

                    embs = embs.clone().detach()
                    emb = self.attn_pool(embs)  # attention pooling
            batch_embs.append(emb)
        return torch.stack(batch_embs, dim=0)  # (batch_size, embed_dim)

    def _roberta_encode_field(self, texts_list):
        """
        texts_list: List[List[str]] (batch)
        returns: (batch_size, embed_dim)
        """
        batch_embs = []
        for texts in texts_list:
            valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
            if len(valid_texts) == 0:
                emb = torch.zeros(self.embed_dim, device=self.device)
            else:
                embs = self.embed_model(valid_texts)  # (num_texts, embed_dim)
                emb = self.attn_pool(embs)  # pooled (embed_dim,)
            batch_embs.append(emb)
        return torch.stack(batch_embs, dim=0)

    def forward(self, batch):
        # batch["code"] is already List[str]

        if self.config["use_roberta_for_emb"]:
            just_emb = self._roberta_encode_field(batch["justification"])
            mon_emb = self._roberta_encode_field(batch["monitoring"])
            eval_emb = self._roberta_encode_field(batch["evaluation"])
            assm_emb = self._roberta_encode_field(batch["assessment"])
            trt_emb = self._roberta_encode_field(batch["treatment"])
            code_emb = self.embed_model(batch["code"])
        else:
            code_emb = self.embed_model.encode(batch["code"], convert_to_tensor=True, device=self.device)
            just_emb = self._encode_field(batch["justification"])
            mon_emb = self._encode_field(batch["monitoring"])
            eval_emb = self._encode_field(batch["evaluation"])
            assm_emb = self._encode_field(batch["assessment"])
            trt_emb = self._encode_field(batch["treatment"])

        # Concatenate all fields
        x = torch.cat([code_emb, just_emb, mon_emb, eval_emb, assm_emb, trt_emb], dim=-1)
        return self.mlp(x)
