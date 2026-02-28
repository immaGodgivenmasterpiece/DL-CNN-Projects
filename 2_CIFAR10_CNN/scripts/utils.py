import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                    # 2_CIFAR10_CNN/
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def get_device():
    """Detect best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluate model on a dataset. Returns (avg_loss, accuracy%)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return running_loss / len(loader), 100. * correct / total


def get_dataloaders(batch_size=64, num_workers=2):
    """
    Returns train_loader and test_loader for CIFAR-10.
    Normalizes using CIFAR-10 channel-wise mean and std.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # R, G, B channel means
            std =(0.2470, 0.2435, 0.2616)   # R, G, B channel stds
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True,  download=True,  transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=False, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_dataset):,} samples | Test: {len(test_dataset):,} samples")
    return train_loader, test_loader


def denormalize(tensor):
    """Reverse CIFAR-10 normalization for visualization."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def show_samples(dataset):
    """Show one sample image per class."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("CIFAR-10: sample by class", fontsize=14)

    shown = {}
    idx = 0
    while len(shown) < 10:
        img, label = dataset[idx]
        if label not in shown:
            shown[label] = (img, label)
        idx += 1

    for i, (label, (img, lbl)) in enumerate(sorted(shown.items())):
        ax = axes[i // 5][i % 5]
        ax.imshow(denormalize(img).permute(1, 2, 0))  # (C,H,W) -> (H,W,C)
        ax.set_title(CLASS_NAMES[lbl])
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_history(train_losses, test_losses, train_accs, test_accs, save_path=None):
    """Plot loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, test_losses,  label='Test Loss',  marker='o')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

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


# ── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(model, loader, device, save_path=None):
    """
    Build and display a confusion matrix.
    Rows = true labels, Columns = predicted labels.
    """
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=int)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs   = model(images)
            _, preds  = outputs.max(1)
            for t, p in zip(labels.numpy(), preds.cpu().numpy()):
                cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
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
        print(f"  {CLASS_NAMES[true]} → {CLASS_NAMES[pred]} : {count} times")


# ── Wrong Prediction Gallery ─────────────────────────────────────────────────

def plot_wrong_predictions(model, loader, device, n=16, save_path=None):
    """
    Collect up to n wrong predictions and display them in a grid.
    Title of each subplot: 'T:{true}  P:{pred}'
    """
    wrong_images = []
    wrong_true   = []
    wrong_pred   = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs  = model(images)
            _, preds = outputs.max(1)

            # Compare on CPU
            preds_cpu = preds.cpu()
            mask      = preds_cpu.ne(labels)

            wrong_images.append(images[mask].cpu())
            wrong_true.append(labels[mask])
            wrong_pred.append(preds_cpu[mask])

            if sum(len(w) for w in wrong_images) >= n:
                break

    wrong_images = torch.cat(wrong_images)[:n]
    wrong_true   = torch.cat(wrong_true)[:n]
    wrong_pred   = torch.cat(wrong_pred)[:n]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i, ax in enumerate(axes.flat):
        if i < len(wrong_images):
            img = denormalize(wrong_images[i]).permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
            ax.imshow(img)
            ax.set_title(f"T:{CLASS_NAMES[wrong_true[i].item()]}\nP:{CLASS_NAMES[wrong_pred[i].item()]}",
                         fontsize=9, color='red')
        ax.axis('off')

    plt.suptitle('Wrong Predictions', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()