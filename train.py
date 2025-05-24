import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from models.moe import MoE
from models.losses import URLoss
from utils.dataset import load_dataset
from utils.metrics import evaluate_metrics, print_metrics, evaluate_bp_standards
from utils.early_stopping import EarlyStopping

def train_one_epoch(model, optimizer, data_loader, device, url_loss):
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to use
        url_loss: Uncertainty regression loss
        
    Returns:
        Average loss, predictions, and labels
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        # Reshape data for model input
        batch_size = data.shape[0]
        data_flat = data.view(batch_size, -1)
        
        # Forward pass
        optimizer.zero_grad()
        output, balance_loss = model(data_flat)
        
        # Compute task losses
        task_loss, task_losses = url_loss(output, target.view(batch_size, -1))
        
        # Combined loss
        loss = task_loss + balance_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.append(output.detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())
    
    # Compute average loss
    avg_loss = total_loss / len(data_loader)
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return avg_loss, all_preds, all_labels

def evaluate(model, data_loader, device, url_loss):
    """
    Evaluate model on validation set
    
    Args:
        model: Model to evaluate
        data_loader: Validation data loader
        device: Device to use
        url_loss: Uncertainty regression loss
        
    Returns:
        Average loss, predictions, and labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Reshape data for model input
            batch_size = data.shape[0]
            data_flat = data.view(batch_size, -1)
            
            # Forward pass
            output, balance_loss = model(data_flat)
            
            # Compute task losses
            task_loss, _ = url_loss(output, target.view(batch_size, -1))
            
            # Combined loss
            loss = task_loss + balance_loss
            
            # Track metrics
            total_loss += loss.item()
            all_preds.append(output.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())
    
    # Compute average loss
    avg_loss = total_loss / len(data_loader)
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return avg_loss, all_preds, all_labels

def main(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    
    # Set up tensorboard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Load dataset
    data_info = load_dataset(
        args.ppg_file, 
        args.labels_file,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        random_state=args.seed
    )
    train_loader = data_info['train_loader']
    val_loader = data_info['val_loader']
    test_loader = data_info['test_loader']
    
    # Define task indices for URLoss
    task_indices = {
        'SBP': (0, 1),  # Systolic Blood Pressure
        'MAP': (1, 2),  # Mean Arterial Pressure
        'DBP': (2, 3),  # Diastolic Blood Pressure
        'HR': (3, 4)    # Heart Rate
    }
    
    # Initialize model
    input_size = data_info['input_size']
    output_size = 4  # SBP, MAP, DBP, HR
    
    model = MoE(
        input_size=input_size,
        output_size=output_size,
        num_experts=args.num_experts,
        hidden_channels=args.hidden_channels,
        noisy_gating=args.noisy_gating,
        noise_type=args.noise_type,
        k=args.k,
        expert_type=args.expert_type,
        use_msfgm=args.use_msfgm
    )
    
    # Initialize weights
    def weights_init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(weights_init)
    model = model.to(device)
    
    # Initialize URLoss
    url_loss = URLoss(task_indices=task_indices).to(device)
    
    # Set up optimizer and scheduler
    optimizer_params = [p for p in model.parameters() if p.requires_grad] + [p for p in url_loss.parameters() if p.requires_grad]
    optimizer = optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=0,
        path=os.path.join(args.output_dir, "weights", "checkpoint.pt")
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        start_time = time.time()
        train_loss, train_preds, train_labels = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            url_loss=url_loss
        )
        train_time = time.time() - start_time
        
        # Validate
        val_loss, val_preds, val_labels = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            url_loss=url_loss
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "weights", "best_model.pt"))
            best_epoch = epoch
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {train_time:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "weights", "best_model.pt")))
    
    # Evaluate on test set
    test_loss, test_preds, test_labels = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        url_loss=url_loss
    )
    
    # Reshape predictions and labels for evaluation
    test_preds = test_preds.reshape(-1, output_size)
    test_labels = test_labels.reshape(-1, output_size)
    
    # Evaluate per task
    task_names = ['SBP', 'MAP', 'DBP', 'HR']
    for i, task in enumerate(task_names):
        metrics = evaluate_metrics(test_labels[:, i], test_preds[:, i])
        print(f"\n--- {task} Metrics ---")
        print_metrics(metrics)
        
        if task in ['SBP', 'MAP', 'DBP']:
            standards = evaluate_bp_standards(metrics)
            print(f"\n--- {task} Standards Compliance ---")
            print(f"IEEE: {standards['IEEE']}")
            print(f"AAMI: {standards['AAMI']}")
            print(f"BHS: {standards['BHS']}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plots", "loss_curves.png"))
    
    # Plot scatter plots for each task
    plt.figure(figsize=(16, 12))
    for i, task in enumerate(task_names):
        plt.subplot(2, 2, i+1)
        plt.scatter(test_labels[:, i], test_preds[:, i], alpha=0.5)
        
        # Add identity line
        min_val = min(test_labels[:, i].min(), test_preds[:, i].min())
        max_val = max(test_labels[:, i].max(), test_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel(f'True {task}')
        plt.ylabel(f'Predicted {task}')
        plt.title(f'{task} Predictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plots", "prediction_scatter.png"))
    
    print("\nTraining completed successfully!")
    print(f"Best model saved at: {os.path.join(args.output_dir, 'weights', 'best_model.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSM²oE model for hemodynamic parameter estimation")
    
    # Data parameters
    parser.add_argument('--ppg_file', type=str, required=True, help='Path to PPG signals file')
    parser.add_argument('--labels_file', type=str, required=True, help='Path to labels file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of remaining data for validation')
    
    # Model parameters
    parser.add_argument('--num_experts', type=int, default=16, help='Number of experts in MoE')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels in expert networks')
    parser.add_argument('--noisy_gating', action='store_true', help='Use noise in gating network')
    parser.add_argument('--noise_type', type=str, default='uniform', 
                      choices=['gaussian', 'uniform', 'poisson', 'beta', 'salt_and_pepper', 'speckle'],
                      help='Type of noise for gating network')
    parser.add_argument('--k', type=int, default=3, help='Number of experts to select for each input')
    parser.add_argument('--expert_type', type=str, default='resnet', choices=['resnet', 'msfgm'],
                      help='Type of expert network')
    parser.add_argument('--use_msfgm', action='store_true', help='Use MSFGM in ResNet blocks')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Miscellaneous
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    args = parser.parse_args()
    main(args)