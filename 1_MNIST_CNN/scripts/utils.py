"""
utils.py
--------
Training utilities: train / evaluate functions,
and all visualisation helpers (loss curves, confusion matrix,
wrong-prediction gallery).
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────
# 1. Train / Evaluate
# ─────────────────────────────────────────

def train(model, loader, criterion, optimizer):
    """Run one full epoch of training. Returns (avg_loss, accuracy %)."""
    model.train()          # activate dropout
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        optimizer.zero_grad()                  # clear gradients from previous batch
        outputs = model(images)                # forward pass  -> (N, 10) logits
        loss    = criterion(outputs, labels)   # CrossEntropyLoss
        loss.backward()                        # backprop
        optimizer.step()                       # weight update

        running_loss += loss.item()
        _, predicted  = outputs.max(1)         # index of highest logit = predicted digit
        correct       += predicted.eq(labels).sum().item()
        total         += labels.size(0)

    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion):
    """Evaluate model on loader. Returns (avg_loss, accuracy %).
    No gradient computation -> faster & less memory."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)

    return running_loss / len(loader), 100.0 * correct / total


# ─────────────────────────────────────────
# 2. Plot Loss & Accuracy Curves
# ─────────────────────────────────────────

def plot_history(train_losses, test_losses, train_accs, test_accs, save_path=None):
    """
    Plot train/test loss and accuracy curves side by side.

    Parameters
    ----------
    train_losses, test_losses : list of float
    train_accs,  test_accs   : list of float
    save_path                : str or None  (e.g. 'results/loss_acc.png')
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, test_losses,  label='Test Loss',  marker='o')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, train_accs, label='Train Acc', marker='o')
    ax2.plot(epochs, test_accs,  label='Test Acc',  marker='o')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────
# 3. Confusion Matrix
# ─────────────────────────────────────────

def plot_confusion_matrix(model, loader, class_names=None, save_path=None):
    """
    Build and display a confusion matrix.
    Rows = true labels, Columns = predicted labels.
    The brighter the cell, the more predictions fell there.

    Parameters
    ----------
    class_names : list of str, e.g. ['0','1',...,'9']
    save_path   : str or None
    """
    num_classes = 10
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm = np.zeros((num_classes, num_classes), dtype=int)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs   = model(images)
            _, preds  = outputs.max(1)
            for t, p in zip(labels.numpy(), preds.numpy()):
                cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    # Annotate each cell with the count
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i][j]),
                    ha='center', va='center',
                    color='white' if cm[i][j] > thresh else 'black',
                    fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()

    # Print most-confused pairs
    print("\nTop confused pairs (true → predicted):")
    off_diag = [(cm[i][j], i, j) for i in range(num_classes)
                                  for j in range(num_classes) if i != j]
    for count, true, pred in sorted(off_diag, reverse=True)[:5]:
        print(f"  {true} → {pred} : {count} times")


# ─────────────────────────────────────────
# 4. Wrong Prediction Gallery
# ─────────────────────────────────────────

def plot_wrong_predictions(model, loader, n=16, save_path=None):
    """
    Collect up to n wrong predictions and display them in a grid.
    Title of each subplot: 'T:{true}  P:{pred}'

    Parameters
    ----------
    n         : int   – how many wrong images to show
    save_path : str or None
    """
    wrong_images = []
    wrong_true   = []
    wrong_pred   = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs  = model(images)
            _, preds = outputs.max(1)
            mask     = preds.ne(labels)          # True where prediction is wrong

            wrong_images.append(images[mask])
            wrong_true.append(labels[mask])
            wrong_pred.append(preds[mask])

            if sum(len(w) for w in wrong_images) >= n:
                break

    wrong_images = torch.cat(wrong_images)[:n]
    wrong_true   = torch.cat(wrong_true)[:n]
    wrong_pred   = torch.cat(wrong_pred)[:n]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

    for i, ax in enumerate(axes.flat):
        if i < len(wrong_images):
            img = wrong_images[i].squeeze().numpy()
            # Denormalise from [-1,1] back to [0,1] for display
            img = (img * 0.5) + 0.5
            ax.imshow(img, cmap='gray')
            ax.set_title(f"T:{wrong_true[i].item()}  P:{wrong_pred[i].item()}",
                         fontsize=9, color='red')
        ax.axis('off')

    plt.suptitle('Wrong Predictions', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()
