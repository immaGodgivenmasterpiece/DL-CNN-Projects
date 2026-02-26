"""
train.py
--------
End-to-end training script for MNIST CNN.

Usage (from project root):
    python scripts/train.py            # fresh training (random init)
    python scripts/train.py --resume   # continue from best_model.pth
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleCNN
from utils import train, evaluate, plot_history, plot_confusion_matrix, plot_wrong_predictions


# ─────────────────────────────────────────
# Args
# ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true',
                    help='Resume training from best_model.pth instead of random init')
args = parser.parse_args()


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 0.001
PATIENCE    = 3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Data
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # pixel → [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True,  transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


# ─────────────────────────────────────────
# Model / Loss / Optimizer
# ─────────────────────────────────────────
model     = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_model_path = os.path.join(RESULTS_DIR, 'best_model.pth')

if args.resume and os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"Resumed from {best_model_path}")
else:
    if args.resume:
        print("No checkpoint found, starting fresh.")
    print("Training from random initialization.")


# ─────────────────────────────────────────
# Training loop with Early Stopping
# ─────────────────────────────────────────
best_acc      = 0.0
counter       = 0
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

for epoch in range(EPOCHS):
    tr_loss, tr_acc = train(model, train_loader, criterion, optimizer)
    te_loss, te_acc = evaluate(model, test_loader, criterion)

    train_losses.append(tr_loss);  test_losses.append(te_loss)
    train_accs.append(tr_acc);     test_accs.append(te_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
    print(f"  Test  Loss: {te_loss:.4f} | Test  Acc: {te_acc:.2f}%")

    if te_acc > best_acc:
        best_acc = te_acc
        counter  = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved! (acc: {best_acc:.2f}%)")
    else:
        counter += 1
        print(f"  patience: {counter}/{PATIENCE}")
        if counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}. Best Test Acc: {best_acc:.2f}%")
            break

# Load best weights
model.load_state_dict(torch.load(best_model_path, weights_only=True))
print("Best model loaded!")

# Save loss/acc curves
plot_history(
    train_losses, test_losses,
    train_accs,   test_accs,
    save_path=os.path.join(RESULTS_DIR, 'loss_acc_curves.png')
)

# Save confusion matrix
plot_confusion_matrix(
    model, test_loader,
    save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png')
)

# Save wrong predictions gallery
plot_wrong_predictions(
    model, test_loader, n=16,
    save_path=os.path.join(RESULTS_DIR, 'wrong_predictions.png')
)
