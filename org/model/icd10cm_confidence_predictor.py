import torch
from torch import nn

import torch
import torch.nn as nn
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


class ICDConfidenceModel(nn.Module):
    def __init__(self, embed_model, device, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.device = device
        self.embed_model = embed_model
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

    def forward(self, batch):
        # batch["code"] is already List[str]
        code_emb = self.embed_model.encode(batch["code"], convert_to_tensor=True, device=self.device)

        just_emb = self._encode_field(batch["justification"])
        mon_emb = self._encode_field(batch["monitoring"])
        eval_emb = self._encode_field(batch["evaluation"])
        assm_emb = self._encode_field(batch["assessment"])
        trt_emb = self._encode_field(batch["treatment"])

        # Concatenate all fields
        x = torch.cat([code_emb, just_emb, mon_emb, eval_emb, assm_emb, trt_emb], dim=-1)
        return self.mlp(x)
