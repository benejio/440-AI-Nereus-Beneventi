# train_dogpoop_mobilenetv2.py
"""
MobileNetV2 training pipeline for binary posture classification:
- Data: ImageFolder train/val/test
- Aug: stronger on train, center-crop eval
- Opt: AdamW + cosine anneal, AMP, early stop
- Outputs: best .pt, training curve, tau-sweep, confusion matrix, PR curve
"""

import os
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from PIL import ImageFile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import itertools
from pathlib import Path
from matplotlib.colors import Normalize

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate a few bad images

# ---------- HARD-CODED PATHS (Windows) ----------
# Root dataset directory with subfolders: train/, val/, test/
DATA_DIR = Path(
    r"C:\Users\Jesse B\OneDrive\Documents\2025 Fall\CPTS 580 - Computer Vision\DogPoopingImages\dpd2024"
)

# Where plots are written
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Best-performing weights checkpoint (state_dict) saved here
SAVE_PATH = Path(
    r"C:\Users\Jesse B\OneDrive\Documents\2025 Fall\CPTS 580 - Computer Vision\mobilenetv2_dogpoop.pt"
)

# ---------- CONFIG ----------
BATCH_TRAIN = 64      # training batch size
BATCH_EVAL = 128      # eval batch size (val/test)
EPOCHS = 50           # max epochs (early-stop may end earlier)
LR = 3e-4             # base LR for AdamW
WD = 1e-4             # weight decay (L2)
NUM_WORKERS = 4       # DataLoader workers (set 0 on Windows if issues)


def get_loaders():
    """
    Build train/val/test DataLoaders with proper transforms.

    Returns:
        (tr, va, ta, classes):
        tr (DataLoader): training loader with strong aug.
        va (DataLoader): validation loader (center-crop).
        ta (DataLoader): test loader (center-crop).
        classes (List[str]): class names from ImageFolder order.
    """
    tx_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tx_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), tx_train)
    val_ds = datasets.ImageFolder(str(DATA_DIR / "val"), tx_eval)
    test_ds = datasets.ImageFolder(str(DATA_DIR / "test"), tx_eval)

    # Pin memory if CUDA available to speed host→device transfers
    pin = torch.cuda.is_available()
    tr = DataLoader(
        train_ds,
        batch_size=BATCH_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    va = DataLoader(
        val_ds,
        batch_size=BATCH_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    ta = DataLoader(
        test_ds,
        batch_size=BATCH_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    return tr, va, ta, train_ds.classes


def build_model(device):
    """
    Create MobileNetV2, freeze features except last block, replace head with 2-way classifier.

    Args:
        device (str): "cuda" or "cpu"

    Returns:
        nn.Module: model moved to device, ready for fine-tuning.
    """
    # Prefer enum weights if available, else fallback to string
    try:
        from torchvision.models import MobileNet_V2_Weights
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        model = mobilenet_v2(weights="IMAGENET1K_V1")

    # Freeze most features for efficiency/stability
    for p in model.features.parameters():
        p.requires_grad = False

    # Unfreeze the last block for task-specific adaptation
    for p in model.features[-1].parameters():
        p.requires_grad = True

    # Replace classifier head: logits over 2 classes (index 1 == "poop")
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)
    return model


@torch.no_grad()
def eval_loader(model, loader, device):
    """
    Evaluate on a loader with simple TTA (horizontal flip), return accuracy and scores.

    Args:
        model (nn.Module): trained or current model (eval mode set inside).
        loader (DataLoader): val/test loader.
        device (str): "cuda" or "cpu".

    Returns:
        Tuple[float, List[float], List[int]]:
        acc: top-1 accuracy
        probs: P(class==1) per sample (after softmax + TTA avg)
        ytrue: ground-truth labels per sample
    """
    model.eval()
    correct = total = 0
    probs, ytrue = [], []
    hflip = transforms.RandomHorizontalFlip(p=1.0)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # forward + TTA (original + flipped)
        p1 = torch.softmax(model(x), 1)
        x2 = hflip(x.detach().cpu()).to(device)
        p2 = torch.softmax(model(x2), 1)
        p = (p1 + p2) / 2

        # accumulate metrics
        pred = p.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        probs.extend(p[:, 1].detach().cpu().tolist())  # class-1 probability
        ytrue.extend(y.detach().cpu().tolist())

    acc = correct / total if total else float("nan")
    return acc, probs, ytrue


def metrics_at_tau(ps, ys, tau):
    """
    Compute precision/recall/F1 at decision threshold tau for class-1.

    Args:
        ps (np.ndarray|List[float]): probabilities for class-1.
        ys (np.ndarray|List[int]): ground-truth labels (0/1).
        tau (float): decision threshold in [0,1].

    Returns:
        Tuple[float,float,float,int,int,int]:
        precision, recall, f1, tp, fp, fn
    """
    yhat = (ps >= tau).astype(int)
    tp = int(((yhat == 1) & (ys == 1)).sum())
    fp = int(((yhat == 1) & (ys == 0)).sum())
    fn = int(((yhat == 0) & (ys == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn


def train():
    """
    Full training loop:
    - Build data, model, optimizer/scheduler, AMP scaler
    - Train with early stopping on val accuracy (patience=6)
    - Save best state_dict to SAVE_PATH
    - Produce figures: training_curve.png, tau_sweep.png, confmat.png, pr_curve.png
    - Print validation-derived tau* and test metrics at tau*
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tr, va, ta, classes = get_loaders()
    print("Classes:", classes)

    model = build_model(device)

    # Loss/opt/sched
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-6)

    # Mixed precision on CUDA
    use_amp = (device == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # Per-epoch logs for plotting
    hist_epochs, hist_train_loss, hist_val_acc, hist_lr = [], [], [], []

    best_val, patience, bad = -1.0, 6, 0
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0

        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * y.size(0)

        # Validation accuracy for early stopping / model selection
        val_acc, _, _ = eval_loader(model, va, device)
        print(f"epoch {epoch}: train_loss={running/len(tr.dataset):.4f} val_acc={val_acc:.3f}")

        # Log stats for curve
        this_lr = opt.param_groups[0]["lr"]
        hist_epochs.append(epoch)
        hist_train_loss.append(running / len(tr.dataset))
        hist_val_acc.append(val_acc)
        hist_lr.append(this_lr)

        # Keep best checkpoint by val_acc
        if val_acc > best_val:
            best_val, bad = val_acc, 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" saved best to {SAVE_PATH}")
        else:
            bad += 1

        # LR schedule & early stop
        sched.step()
        if bad >= patience:
            print("Early stop.")
            break

    # ---- Load best model ----
    if SAVE_PATH.exists():
        state = torch.load(SAVE_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)

    # ---- Save training curve figure ----
    plt.figure(figsize=(7, 4.2))
    ax1 = plt.gca()
    ln1 = ax1.plot(hist_epochs, hist_train_loss, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")

    ax2 = ax1.twinx()
    ln2 = ax2.plot(hist_epochs, hist_val_acc, linestyle="--", label="Val accuracy")
    ax2.set_ylabel("Val accuracy")

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    ax1.grid(True, alpha=0.3)
    plt.title("MobileNetV2 training: loss & validation accuracy")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "training_curve.png", dpi=220)
    plt.close()

    # ---- Tau search on validation ----
    val_acc, val_probs, val_true = eval_loader(model, va, device)
    print(f"VAL accuracy (for tau search): {val_acc:.3f}")

    ys_val = np.array(val_true, dtype=int)
    ps_val = np.array(val_probs, dtype=float)

    # Sweep tau in [0.30, 0.80] in 0.01 increments
    grid = np.linspace(0.30, 0.80, 51)
    best_tau, best_f1, best_stats = 0.50, -1.0, None
    f1s = []
    for tau in grid:
        prec, rec, f1, tp, fp, fn = metrics_at_tau(ps_val, ys_val, tau)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1, best_tau, best_stats = f1, float(tau), (prec, rec, tp, fp, fn)

    # tau-sweep figure
    plt.figure(figsize=(6.5, 3.6))
    plt.plot(grid, f1s, label="F1 vs tau")
    plt.axvline(best_tau, linestyle="--", label=f"tau* = {best_tau:.2f}")
    plt.xlabel("Decision threshold tau")
    plt.ylabel("F1")
    plt.title("Validation F1 across thresholds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "tau_sweep.png", dpi=220)
    plt.close()

    print(
        f"Selected tau={best_tau:.2f} on VAL (F1={best_f1:.3f}, "
        f"P={best_stats[0]:.3f}, R={best_stats[1]:.3f}, "
        f"tp={best_stats[2]}, fp={best_stats[3]}, fn={best_stats[4]})"
    )

    # ---- Evaluate test at selected tau (and short sweep) ----
    test_acc, test_probs, ytrue = eval_loader(model, ta, device)
    print(f"TEST accuracy: {test_acc:.3f}")

    if ytrue:
        ys = np.array(ytrue, dtype=int)
        ps = np.array(test_probs, dtype=float)

        # Report at tau*
        prec, rec, f1, tp, fp, fn = metrics_at_tau(ps, ys, best_tau)
        print(f"TEST @ tau={best_tau:.2f}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
              f"(tp={tp}, fp={fp}, fn={fn})")

        # Short sweep (fixed tau set) for paper consistency
        for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            p2, r2, f12, tp2, fp2, fn2 = metrics_at_tau(ps, ys, tau)
            print(f"tau={tau:.1f}: P={p2:.3f} R={r2:.3f} F1={f12:.3f} "
                  f"(tp={tp2}, fp={fp2}, fn={fn2})")

        # ---- Confusion matrix at tau* ----
        yhat = (ps >= best_tau).astype(int)
        cm = confusion_matrix(ys, yhat, labels=[0, 1])  # [[tn, fp],[fn, tp]]

        plt.figure(figsize=(5, 4.4))
        ax = plt.gca()

        # Heatmap with nicer palette
        norm = Normalize(vmin=0, vmax=cm.max())
        im = ax.imshow(cm, cmap="YlGnBu", norm=norm)

        # Cosmetics
        ax.set_title(f"Confusion Matrix @ tau*={best_tau:.2f}")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        classes = ["notpoop", "poop"]
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # minor grid for separation
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Annotated counts with auto-contrast text color
        for (i, j), val in np.ndenumerate(cm):
            text_color = "white" if norm(val) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "confmat.png", dpi=220)
        plt.close()

        # ---- Precision-Recall curve + marked tau* point ----
        ap = average_precision_score(ys, ps)
        precisions, recalls, _ = precision_recall_curve(ys, ps)

        # exact PR at tau*
        yhat = (ps >= best_tau).astype(int)
        tp_ = int(((yhat == 1) & (ys == 1)).sum())
        fp_ = int(((yhat == 1) & (ys == 0)).sum())
        fn_ = int(((yhat == 0) & (ys == 1)).sum())
        prec_star = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        rec_star = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0

        plt.figure(figsize=(6.2, 4.0))
        ax = plt.gca()
        ax.plot(recalls, precisions, label=f"PR (AP={ap:.3f})", linewidth=2)
        ax.scatter([rec_star], [prec_star], s=60, zorder=3, label=f"tau*={best_tau:.2f}")

        # zoom near (1,1)
        ax.set_xlim(0.90, 1.10)
        ax.set_ylim(0.90, 1.10)
        ax.set_xticks(np.linspace(0.90, 1.00, 6))
        ax.set_yticks(np.linspace(0.90, 1.00, 6))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (zoomed)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "pr_curve.png", dpi=220)
        plt.close()

    return model


if __name__ == "__main__":
    # For Windows multiprocessing safety
    import multiprocessing as mp
    mp.freeze_support()
    train()