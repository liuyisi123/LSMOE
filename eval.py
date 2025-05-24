import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.moe import MoE
from models.losses import URLoss
from utils.dataset import load_dataset
from utils.metrics import evaluate_metrics, print_metrics, evaluate_bp_standards

def evaluate_model(model, data_loader, device, url_loss):
    """
    Evaluate model on given dataset
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        url_loss: Uncertainty regression loss
        
    Returns:
        Average loss, predictions, labels, and uncertainty estimates
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Reshape data for model input
            batch_size = data.shape[0]
            data_flat = data.view(batch_size, -1)
            
            # Forward pass
            output, balance_loss = model(data_flat)
            
            # Compute task losses
            task_loss, task_losses = url_loss(output, target.view(batch_size, -1))
            
            # Combined loss
            loss = task_loss + balance_loss
            
            # Extract predictions and uncertainties
            # Assuming output format: [pred1, sigma1, pred2, sigma2, ...]
            preds = output[:, ::2]  # Even indices are predictions
            uncertainties = output[:, 1::2]  # Odd indices are uncertainties
            
            # Track metrics
            total_loss += loss.item()
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())
            all_uncertainties.append(uncertainties.detach().cpu().numpy())
    
    # Compute average loss
    avg_loss = total_loss / len(data_loader)
    
    # Concatenate predictions, labels, and uncertainties
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    
    return avg_loss, all_preds, all_labels, all_uncertainties

def plot_results(labels, preds, uncertainties, task_names, output_dir):
    """
    Create visualization plots for evaluation results
    
    Args:
        labels: True labels
        preds: Predictions
        uncertainties: Uncertainty estimates
        task_names: Names of tasks
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "eval_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Scatter plots with uncertainty
    plt.figure(figsize=(16, 12))
    for i, task in enumerate(task_names):
        plt.subplot(2, 2, i+1)
        scatter = plt.scatter(labels[:, i], preds[:, i], 
                            c=uncertainties[:, i], cmap='viridis', alpha=0.6)
        
        # Add identity line
        min_val = min(labels[:, i].min(), preds[:, i].min())
        max_val = max(labels[:, i].max(), preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel(f'True {task}')
        plt.ylabel(f'Predicted {task}')
        plt.title(f'{task} Predictions (colored by uncertainty)')
        plt.colorbar(scatter, label='Uncertainty')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "prediction_scatter_with_uncertainty.png"), dpi=300)
    plt.close()
    
    # Error vs Uncertainty plots
    plt.figure(figsize=(16, 12))
    for i, task in enumerate(task_names):
        plt.subplot(2, 2, i+1)
        errors = np.abs(labels[:, i] - preds[:, i])
        plt.scatter(uncertainties[:, i], errors, alpha=0.5)
        
        plt.xlabel(f'{task} Uncertainty')
        plt.ylabel(f'{task} Absolute Error')
        plt.title(f'{task}: Error vs Uncertainty')
        
        # Add correlation coefficient
        corr = np.corrcoef(uncertainties[:, i], errors)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_vs_uncertainty.png"), dpi=300)
    plt.close()
    
    # Bland-Altman plots
    plt.figure(figsize=(16, 12))
    for i, task in enumerate(task_names):
        plt.subplot(2, 2, i+1)
        mean_values = (labels[:, i] + preds[:, i]) / 2
        differences = preds[:, i] - labels[:, i]
        
        plt.scatter(mean_values, differences, alpha=0.5)
        
        # Add mean difference line
        mean_diff = np.mean(differences)
        plt.axhline(y=mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.2f}')
        
        # Add limits of agreement
        std_diff = np.std(differences)
        plt.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1, 
                   label=f'+1.96SD: {mean_diff + 1.96*std_diff:.2f}')
        plt.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1,
                   label=f'-1.96SD: {mean_diff - 1.96*std_diff:.2f}')
        
        plt.xlabel(f'Mean of True and Predicted {task}')
        plt.ylabel(f'Predicted - True {task}')
        plt.title(f'{task} Bland-Altman Plot')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "bland_altman_plots.png"), dpi=300)
    plt.close()

def save_results(labels, preds, uncertainties, metrics_dict, task_names, output_dir):
    """
    Save evaluation results to files
    
    Args:
        labels: True labels
        preds: Predictions  
        uncertainties: Uncertainty estimates
        metrics_dict: Dictionary of metrics for each task
        task_names: Names of tasks
        output_dir: Directory to save results
    """
    results_dir = os.path.join(output_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions and labels
    np.save(os.path.join(results_dir, "true_labels.npy"), labels)
    np.save(os.path.join(results_dir, "predictions.npy"), preds)
    np.save(os.path.join(results_dir, "uncertainties.npy"), uncertainties)
    
    # Save metrics summary
    with open(os.path.join(results_dir, "metrics_summary.txt"), 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for task in task_names:
            if task in metrics_dict:
                f.write(f"--- {task} Metrics ---\n")
                metrics = metrics_dict[task]
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{metric_name}: {value:.4f}\n")
                    else:
                        f.write(f"{metric_name}: {value}\n")
                f.write("\n")
                
                # Add standards compliance for BP tasks
                if task in ['SBP', 'MAP', 'DBP']:
                    standards = evaluate_bp_standards(metrics)
                    f.write(f"--- {task} Standards Compliance ---\n")
                    f.write(f"IEEE: {standards['IEEE']}\n")
                    f.write(f"AAMI: {standards['AAMI']}\n")
                    f.write(f"BHS: {standards['BHS']}\n\n")

def main(args):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for evaluation")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    data_info = load_dataset(
        args.ppg_file, 
        args.labels_file,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        random_state=args.seed
    )
    
    # Select dataset split to evaluate
    if args.eval_split == 'test':
        data_loader = data_info['test_loader']
        print("Evaluating on test set")
    elif args.eval_split == 'val':
        data_loader = data_info['val_loader']
        print("Evaluating on validation set")
    else:
        data_loader = data_info['train_loader']
        print("Evaluating on training set")
    
    # Define task indices for URLoss
    task_indices = {
        'SBP': (0, 1),  # Systolic Blood Pressure
        'MAP': (1, 2),  # Mean Arterial Pressure
        'DBP': (2, 3),  # Diastolic Blood Pressure
        'HR': (3, 4)    # Heart Rate
    }
    
    # Initialize model
    input_size = data_info['input_size']
    output_size = 8  # 4 predictions + 4 uncertainties (SBP, MAP, DBP, HR)
    
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
    
    # Load trained model weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    print(f"Loading model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Initialize URLoss
    url_loss = URLoss(task_indices=task_indices).to(device)
    
    # Evaluate model
    print("Starting evaluation...")
    test_loss, test_preds, test_labels, test_uncertainties = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        url_loss=url_loss
    )
    
    print(f"Evaluation Loss: {test_loss:.4f}")
    
    # Reshape for per-task evaluation
    task_names = ['SBP', 'MAP', 'DBP', 'HR']
    num_tasks = len(task_names)
    
    # Evaluate per task and store metrics
    metrics_dict = {}
    for i, task in enumerate(task_names):
        print(f"\n--- {task} Metrics ---")
        metrics = evaluate_metrics(test_labels[:, i], test_preds[:, i])
        print_metrics(metrics)
        metrics_dict[task] = metrics
        
        # Add uncertainty metrics
        uncertainty_mean = np.mean(test_uncertainties[:, i])
        uncertainty_std = np.std(test_uncertainties[:, i])
        print(f"Mean Uncertainty: {uncertainty_mean:.4f}")
        print(f"Std Uncertainty: {uncertainty_std:.4f}")
        
        # Standards compliance for blood pressure tasks
        if task in ['SBP', 'MAP', 'DBP']:
            standards = evaluate_bp_standards(metrics)
            print(f"\n--- {task} Standards Compliance ---")
            print(f"IEEE: {standards['IEEE']}")
            print(f"AAMI: {standards['AAMI']}")
            print(f"BHS: {standards['BHS']}")
    
    # Create visualizations
    print("\nGenerating visualization plots...")
    plot_results(test_labels, test_preds, test_uncertainties, task_names, args.output_dir)
    
    # Save results
    print("Saving evaluation results...")
    save_results(test_labels, test_preds, test_uncertainties, metrics_dict, task_names, args.output_dir)
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained LSM²oE model for hemodynamic parameter estimation")
    
    # Data parameters
    parser.add_argument('--ppg_file', type=str, required=True, help='Path to PPG signals file')
    parser.add_argument('--labels_file', type=str, required=True, help='Path to labels file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of remaining data for validation')
    parser.add_argument('--eval_split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    
    # Model parameters (should match training configuration)
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
    
    # Evaluation parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Miscellaneous
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for evaluation')
    parser.add_argument('--output_dir', type=str, default='./eval_output', help='Output directory for results')
    
    args = parser.parse_args()
    main(args)