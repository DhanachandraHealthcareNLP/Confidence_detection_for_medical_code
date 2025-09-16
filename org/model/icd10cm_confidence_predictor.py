import torch
from torch import nn


class ICDConfidenceModel(nn.Module):
    def __init__(self, embed_model, device, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.device = device
        self.embed_model = embed_model
        self.embed_dim = embed_model.get_sentence_embedding_dimension()
        input_dim = self.embed_dim * 6  # code + 5 MEAT fields
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dim: {hidden_dim}")
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_field(self, texts_list):
        """
        texts_list: List[List[str]] (batch)
        Returns: (batch_size, embed_dim)
        """
        batch_embs = []
        for texts in texts_list:
            if not texts:
                emb = torch.zeros(self.embed_dim, device=self.device)
            else:
                # flatten in case of nested list
                if isinstance(texts[0], (list, tuple)):
                    texts = [y for x in texts for y in x if isinstance(y, str)]
                embs = self.embed_model.encode(texts, convert_to_tensor=True, device=self.device)
                emb = embs.mean(dim=0)
            batch_embs.append(emb)
        return torch.stack(batch_embs, dim=0)

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
