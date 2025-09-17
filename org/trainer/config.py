CONFIG = {
    "embed_model_name": "sentence-transformers/all-mpnet-base-v2",
    "batch_size": 8,   # smaller batch since we encode inside forward
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "epochs": 32,
    "dropout": 0.2,
    "hidden_dim": 512,
    "val_split": 0.2,
    "seed": 42,
    "save_path": "icd_confidence_model.pt",
    "project": "icd10-confidence",
    "use_roberta_for_emb": True,
    "roberta_model_name": "pminervini/RoBERTa-base-PM-M3-Voc-hf",
    "scheduler": "cosine",
    "warmup_ratio": 0.1,
}