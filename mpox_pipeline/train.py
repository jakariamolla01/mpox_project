import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_builder import build_model, BACKBONES
from data_utils import build_dataframe, get_kfold_splits, make_dataloaders, compute_class_weights
from evaluate import compute_metrics

SEED = 108
BACKBONE_LIST = list(BACKBONES.keys())
ATTENTION_TYPES = ["none", "eca", "se", "cbam"]
DEPTHS = ["single", "double"]

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return compute_metrics(all_labels, all_preds)


def train_one_config(backbone_name, attention_type, depth, df, classes, splits, output_dir, batch_size, epochs, device):
    fold_metrics = []
    
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        model, img_size = build_model(backbone_name, len(classes), attention_type, depth)
        model = model.to(device)

        train_loader, val_loader = make_dataloaders(df, train_idx, test_idx, img_size, batch_size)
        
        class_weights = compute_class_weights(df.iloc[train_idx], len(classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = Adam(model.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
        scaler = torch.amp.GradScaler('cuda')

        best_acc = 0.0
        patience, patience_counter = 8, 0
        ckpt_path = os.path.join(output_dir, f"{backbone_name}_{attention_type}_{depth}_fold{fold_i}.pt")

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            metrics = evaluate_model(model, val_loader, device)
            val_acc = metrics["accuracy"]
            
            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(torch.load(ckpt_path))
        final_metrics = evaluate_model(model, val_loader, device)
        final_metrics["fold"] = fold_i
        fold_metrics.append(final_metrics)
        print(f"[{backbone_name}-{attention_type}-{depth}] fold {fold_i}: acc={final_metrics['accuracy']:.4f}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="PyTorch Mpox Pipeline")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset folder")
    parser.add_argument("--output_dir", type=str, default="results", help="Output folder")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs count")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df, classes = build_dataframe(args.data_dir)
    splits = get_kfold_splits(df, n_splits=5, seed=SEED)

    all_results = []
    for backbone_name in BACKBONE_LIST:
        for attention_type in ATTENTION_TYPES:
            depths = ["single"] if attention_type == "none" else DEPTHS
            for depth in depths:
                fold_metrics = train_one_config(
                    backbone_name, attention_type, depth, df, classes, splits,
                    args.output_dir, args.batch_size, args.epochs, device
                )
                mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
                all_results.append({
                    "backbone": backbone_name, "attention": attention_type, "depth": depth,
                    "mean_accuracy": float(mean_acc), "fold_metrics": fold_metrics,
                })
                with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(all_results, f, indent=2)

    print("Done. Results saved to results/results.json")

if __name__ == "__main__":
    main()