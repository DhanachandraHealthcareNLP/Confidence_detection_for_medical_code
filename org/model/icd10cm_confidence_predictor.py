import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) -> 1 for valid tokens, 0 for padding
        """
        weights = self.attn(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(weights, dim=-1)  # (batch, seq_len)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)


class RobertaEmbedder(nn.Module):
    def __init__(self, model_name="roberta-base", device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size
        self.device = device
        self.to(device)

    def forward(self, texts, max_length=512):
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
        return outputs.last_hidden_state, enc["attention_mask"]

    def get_sentence_embedding_dimension(self):
        return self.embed_dim


class ICDConfidenceModel(nn.Module):
    def __init__(self, embed_model, device, hidden_dim=512, dropout=0.4, config=None):
        super().__init__()
        self.device = device
        self.embed_model = embed_model
        self.config = config

        if self.config["use_roberta_for_emb"]:
            self.embed_model = RobertaEmbedder(self.config["roberta_model_name"], device=device)

        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        self.attn_pool = AttentionPooling(self.embed_dim)

        input_dim = self.embed_dim * 6  # code + 5 MEAT fields

        # Residual MLP head
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

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
                hidden_states, mask = self.embed_model(valid_texts)  # (num_texts, seq_len, hidden_dim)
                emb = self.attn_pool(hidden_states, mask)  # (num_texts, hidden_dim)
                emb = emb.mean(dim=0)  # aggregate across multiple texts
            batch_embs.append(emb)
        return torch.stack(batch_embs, dim=0)

    def _encode_field(self, texts_list):
        """
        For SentenceTransformer (non-RoBERTa) case
        """
        batch_embs = []
        for texts in texts_list:
            valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
            if len(valid_texts) == 0:
                emb = torch.zeros(self.embed_dim, device=self.device)
            else:
                embs = self.embed_model.encode(
                    valid_texts,
                    convert_to_tensor=True,
                    device=self.device
                )  # (num_texts, embed_dim)
                emb = self.attn_pool(embs.unsqueeze(0)).squeeze(0)  # add batch dim
            batch_embs.append(emb)
        return torch.stack(batch_embs, dim=0)

    def forward(self, batch):
        if self.config["use_roberta_for_emb"]:
            just_emb = self._roberta_encode_field(batch["justification"])
            mon_emb = self._roberta_encode_field(batch["monitoring"])
            eval_emb = self._roberta_encode_field(batch["evaluation"])
            assm_emb = self._roberta_encode_field(batch["assessment"])
            trt_emb = self._roberta_encode_field(batch["treatment"])
            code_hidden, code_mask = self.embed_model(batch["code"])
            code_emb = self.attn_pool(code_hidden, code_mask)
        else:
            code_emb = self.embed_model.encode(batch["code"], convert_to_tensor=True, device=self.device)
            just_emb = self._encode_field(batch["justification"])
            mon_emb = self._encode_field(batch["monitoring"])
            eval_emb = self._encode_field(batch["evaluation"])
            assm_emb = self._encode_field(batch["assessment"])
            trt_emb = self._encode_field(batch["treatment"])

        # Concatenate
        x = torch.cat([code_emb, just_emb, mon_emb, eval_emb, assm_emb, trt_emb], dim=-1)

        # Residual MLP
        features = self.mlp(x)
        logits = self.classifier(features + x[:, :features.size(1)])  # add skip connection from part of input
        return logits