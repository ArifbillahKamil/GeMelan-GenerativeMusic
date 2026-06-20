import os
import sys
import json
import math
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.model.lstm import MelodyLSTM

class MelodyDataset(Dataset):

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.tensor(data['X'], dtype=torch.long)
        self.y = torch.tensor(data['y'], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, loader, criterion, optimizer, device, clip_grad=1.0):

    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        hidden = model.init_hidden(X_batch.size(0), device)

        optimizer.zero_grad()

        logits, _ = model(X_batch, hidden)

        loss = criterion(
            logits.view(-1, model.vocab_size),
            y_batch.view(-1)
        )

        # Backward pass
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            hidden = model.init_hidden(X_batch.size(0), device)
            logits, _ = model(X_batch, hidden)

            loss = criterion(
                logits.view(-1, model.vocab_size),
                y_batch.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch':      epoch,
        'val_loss':   val_loss,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'vocab_size':  model.vocab_size,
            'embed_dim':   model.embed_dim,
            'hidden_size': model.hidden_size,
            'num_layers':  model.num_layers,
            'dropout':     model.dropout,
        }
    }, path)


def save_log(log: list, path: str):
    with open(path, 'w') as f:
        json.dump(log, f, indent=2)

def run_training(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("=" * 55)
    print("  GeMelan - Training MelodyLSTM")
    print("=" * 55)
    print(f"  Device       : {device}")
    print(f"  Seed         : {args.seed}")

    with open(os.path.join(args.data_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = vocab['vocab_size']
    pad_idx    = vocab['special']['PAD']
    print(f"  Vocab size   : {vocab_size}")

    train_dataset = MelodyDataset(os.path.join(args.data_dir, 'train.npz'))
    val_dataset   = MelodyDataset(os.path.join(args.data_dir, 'val.npz'))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"  Train size   : {len(train_dataset):,} windows")
    print(f"  Val size     : {len(val_dataset):,} windows")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Steps/epoch  : {len(train_loader)}")

    model = MelodyLSTM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)
    model.summary()

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # Persiapan Output 
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    best_val_loss  = float('inf')
    patience_count = 0
    log = []

    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    log_path        = os.path.join('outputs/logs', 'training_log.json')

    print(f"\n  Epochs       : {args.epochs}")
    print(f"  LR           : {args.lr}")
    print(f"  Early stop   : patience={args.patience}")
    print(f"  Checkpoint   : {checkpoint_path}")
    print("=" * 55)
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Perplexity':>10} | {'LR':>8} | {'Time':>6}")
    print("-" * 65)

    # Training Loop 
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.clip_grad)
        val_loss   = validate_epoch(model, val_loader, criterion, device)

        # Perplexity = exp(loss) — metrik standar untuk language model
        perplexity = math.exp(val_loss)

        # Scheduler step berdasarkan val loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {perplexity:>10.2f} | {current_lr:>8.6f} | {elapsed:>5.1f}s")

        # Log
        log.append({
            'epoch':      epoch,
            'train_loss': round(train_loss, 6),
            'val_loss':   round(val_loss, 6),
            'perplexity': round(perplexity, 4),
            'lr':         current_lr,
            'time_sec':   round(elapsed, 2),
        })
        save_log(log, log_path)

        # Simpan checkpoint terbaik
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"         -> ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  Early stopping triggered (patience={args.patience})")
                break

    print("-" * 65)
    print("\n  Training selesai!")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Best perplexity: {math.exp(best_val_loss):.2f}")
    print(f"  Checkpoint     : {checkpoint_path}")
    print(f"  Log            : {log_path}")
    print("=" * 55)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeMelan - Train MelodyLSTM')

    # Data
    parser.add_argument('--data_dir',      type=str,   default='data/processed')
    parser.add_argument('--checkpoint_dir',type=str,   default='models')

    # Model
    parser.add_argument('--embed_dim',     type=int,   default=64)
    parser.add_argument('--hidden_size',   type=int,   default=256)
    parser.add_argument('--num_layers',    type=int,   default=2)
    parser.add_argument('--dropout',       type=float, default=0.3)

    # Training
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--clip_grad',     type=float, default=1.0)
    parser.add_argument('--patience',      type=int,   default=10)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--cpu',           action='store_true',
                        help='Paksa gunakan CPU meskipun ada GPU')

    args = parser.parse_args()
    run_training(args)