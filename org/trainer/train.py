import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from org.trainer.dataset import collate_fn

use_wandb = False
if use_wandb:
    import wandb

from config import CONFIG
from org.data_processing.read_input_data import read_json_file, read_label_data, read_code_description
from org.model.icd10cm_confidence_predictor import ICDConfidenceModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
from sentence_transformers import SentenceTransformer

print("ALl libraries loaded!")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train"):
        optimizer.zero_grad()
        logits = model(batch)
        labels = torch.tensor(batch["label"], dtype=torch.float32, device=device).unsqueeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_logits, all_labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            logits = model(batch)
            labels = torch.tensor(batch["label"], dtype=torch.float32, device=device).unsqueeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    print(f"Val loss: {total_loss / len(loader.dataset)}")
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    }


# -------------------------
# Main
# -------------------------
def main(tr_datapoints, val_datapoints, collate_fn):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_wandb:
        wandb.login(key="e0b3f587c775832e1793ec717ee83b24a340d7be")
        wandb.init(project=CONFIG["project"], config=CONFIG)
    print("Loading the embedding model!")
    embed_model = SentenceTransformer(CONFIG["embed_model_name"], device=DEVICE)
    print(f"Embedding model dimension: {embed_model.get_sentence_embedding_dimension()}")
    # self, embed_model, device, hidden_dim = 512, dropout = 0.2
    model = ICDConfidenceModel(embed_model, DEVICE, hidden_dim=CONFIG["hidden_dim"], dropout=CONFIG["dropout"]).to(
        DEVICE)
    print("Loaded the embedding model!")
    # Split train/val
    from sklearn.model_selection import train_test_split
    # train_points, val_points = train_test_split(
    #     datapoints,
    #     test_size=CONFIG["val_split"],
    #     random_state=CONFIG["seed"],
    #     stratify=[p[-1] for p in datapoints]
    # )

    train_loader = DataLoader(tr_datapoints, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_datapoints, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    best_val_f1 = 0
    # for epoch in tqdm(range(1, CONFIG["epochs"] + 1), desc="Training Epoch:"):
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device=DEVICE)
        print(f"Training loss: {train_loss}")
        metrics = eval_epoch(model, val_loader, criterion, DEVICE)
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in metrics.items()}
            })
        model_saving_path = f"../checkpoints/icd_confidence_model_{epoch + 1}.pt"
        torch.save(model.state_dict(), model_saving_path)
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            torch.save(model.state_dict(), CONFIG["save_path"])
            if use_wandb:
                wandb.save(CONFIG["save_path"])
            print(f"✅ Saved new best model (F1={best_val_f1:.4f})")

    print("Training done. Best F1:", best_val_f1)


# -------------------------
# Example datapoints
# -------------------------
def load_datapoints_stub():
    return [
        ("I63.512", ["ischemic stroke"], ["patient lethargic"], ["repeat CT"], ["holding aspirin"],
         ["atorvastatin 80mg"], 1),
        ("F32.A", ["Depression"], [], [], ["continue Fluoxetine"], ["Fluoxetine 20mg"], 1),
        ("E11.9", ["T2DM w/o complications"], [], [], [], [], 0),
        ("I63.512", ["ischemic stroke"], ["patient lethargic"], ["repeat CT"], ["holding aspirin"],
         ["atorvastatin 80mg"], 1),
        ("F32.A", ["Depression"], [], [], ["continue Fluoxetine"], ["Fluoxetine 20mg"], 1),
        ("E11.9", ["T2DM w/o complications"], [], [], [], [], 0),
    ]


def get_datapoints(data_file_path, label_info_dict, code_dict, sample_size=100):
    datapoints = []
    with open(data_file_path) as reader:
        next(reader)
        for i, line in enumerate(reader):
            if (i + 1) == sample_size:
                break
            cols = line.split(',')
            cur_datapoint = read_json_file(cols[-1].strip(), label_info_dict[cols[0].strip()], code_dict)
            datapoints.extend(cur_datapoint)

    return datapoints


def calculate_class_distribution(train_data_points, val_data_points):
    def get_distribution(data_points, name=""):
        labels = [dp[-1] for dp in data_points]  # last element is label
        counts = Counter(labels)
        total = sum(counts.values())
        distribution = {cls: f"{cnt} ({cnt / total:.2%})" for cls, cnt in counts.items()}
        # print(f"{name} Distribution: {distribution}")
        return distribution

    train_dist = get_distribution(train_data_points, "Train")
    val_dist = get_distribution(val_data_points, "Validation")

    return train_dist, val_dist


if __name__ == "__main__":
    train_data_file_path = "../data/train_split.csv"
    val_data_file_path = "../data/val_split.csv"

    csv_path = "../data/Sentara Model Training .csv"
    label_info_dict = read_label_data(csv_path)

    code_desc_path = "../data/icd_codes.csv"
    code_dict = read_code_description(code_desc_path)
    train_data_points = get_datapoints(train_data_file_path, label_info_dict, code_dict)
    val_data_points = get_datapoints(val_data_file_path, label_info_dict, code_dict)

    train_dist, val_dist = calculate_class_distribution(train_data_points, val_data_points)
    print(f"Train dist: {train_dist}")
    print(f"Val dist: {val_dist}")

    # datapoints = load_datapoints_stub()
    main(tr_datapoints=train_data_points, val_datapoints=val_data_points, collate_fn=collate_fn)
