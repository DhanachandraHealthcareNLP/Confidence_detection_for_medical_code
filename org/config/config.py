CONFIG = {
    "embed_model_name": "sentence-transformers/all-mpnet-base-v2",
    "batch_size": 8,   # smaller batch since we encode inside forward
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "epochs": 5,
    "dropout": 0.2,
    "hidden_dim": 512,
    "val_split": 0.2,
    "seed": 42,
    "save_path": "icd_confidence_model.pt",
    "project": "icd10-confidence"
}