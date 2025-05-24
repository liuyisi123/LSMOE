import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.moe import MoE
from models.losses import URLoss
from utils.dataset import load_dataset
from utils.metrics import evaluate_metrics, print_metrics, evaluate_bp_standards
from utils.pruning import structured_prune_model

def bidirectional_knowledge_distillation(teacher_model, student_model, data_loader, device, alpha=0.5, beta=0.5, gamma=0.1):
    """
    Perform bidirectional knowledge distillation
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        data_loader: Data loader for distillation
        device: Device to use
        alpha: Weight for hard loss (student predictions vs ground truth)
        beta: Weight for soft loss (student predictions vs teacher predictions)
        gamma: Weight for bidirectional loss
        
    Returns:
        Updated teacher and student models
    """
    # Set models to training mode
    teacher_model.train()
    student_model.train()
    
    # Set up optimizers
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.0001)
    student_optimizer = optim.Adam(student_model.parameters(), lr=0.0005)
    
    # MSE loss for regression
    mse_loss = nn.MSELoss()
    
    # Track metrics
    total_hard_loss = 0
    total_soft_loss = 0
    total_bi_loss = 0
    
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        
        # Reshape data for model input
        batch_size = data.shape[0]
        data_flat = data.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        
        # Forward pass - teacher
        with torch.no_grad():
            teacher_output, _ = teacher_model(data_flat)
        
        # Forward pass - student
        student_output, _ = student_model(data_flat)
        
        # Compute losses
        hard_loss = mse_loss(student_output, target_flat)
        soft_loss = mse_loss(student_output, teacher_output)
        
        # Check where student outperforms teacher
        teacher_error = mse_loss(teacher_output, target_flat, reduction='none')
        student_error = mse_loss(student_output, target_flat, reduction='none')
        improvement_mask = (student_error < teacher_error).float()
        bi_loss = mse_loss(teacher_output, student_output.detach() * improvement_mask)
        
        # Update student
        student_loss = alpha * hard_loss + beta * soft_loss
        student_optimizer.zero_grad()
        student_loss.backward()
        student_optimizer.step()
        
        # Update teacher with bidirectional loss (only where student is better)
        teacher_optimizer.zero_grad()
        bi_loss.backward()
        teacher_optimizer.step()
        
        # Track metrics
        total_hard_loss += hard_loss.item()
        total_soft_loss += soft_loss.item()
        total_bi_loss += bi_loss.item()
    
    # Compute average losses
    avg_hard_loss = total_hard_loss / len(data_loader)
    avg_soft_loss = total_soft_loss / len(data_loader)
    avg_bi_loss = total_bi_loss / len(data_loader)
    
    print(f"Distillation: Hard Loss={avg_hard_loss:.4f}, Soft Loss={avg_soft_loss:.4f}, Bi Loss={avg_bi_loss:.4f}")
    
    return teacher_model, student_model

def evaluate_model_metrics(model, data_loader, device, model_name="Model"):
    """
    Evaluate model performance metrics
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        model_name: Name of model for printing
        
    Returns:
        Average metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Track inference time
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Reshape data for model input
            batch_size = data.shape[0]
            data_flat = data.view(batch_size, -1)
            
            # Forward pass
            output, _ = model(data_flat)
            
            # Track predictions and labels
            all_preds.append(output.cpu().numpy())
            all_labels.append(target.cpu().numpy())
    
    # Compute average inference time
    total_time = time.time() - start_time
    avg_inference_time = total_time / len(data_loader.dataset)
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Reshape to [num_samples, num_tasks]
    output_size = all_preds.shape[1]
    all_preds = all_preds.reshape(-1, output_size)
    all_labels = all_labels.reshape(-1, output_size)
    
    # Evaluate per task
    task_names = ['SBP', 'MAP', 'DBP', 'HR']
    task_metrics = {}
    
    print(f"\n--- {model_name} Performance ---")
    print(f"Average inference time per sample: {avg_inference_time*1000:.2f} ms")
    
    for i, task in enumerate(task_names):
        metrics = evaluate_metrics(all_labels[:, i], all_preds[:, i])
        task_metrics[task] = metrics
        
        print(f"\n{task} Metrics:")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  R2: {metrics['R2']:.4f}")
        
        if task in ['SBP', 'MAP', 'DBP']:
            standards = evaluate_bp_standards(metrics)
            print(f"  Standards: IEEE={standards['IEEE']}, AAMI={standards['AAMI']}, BHS={standards['BHS']}")
    
    return {
        'task_metrics': task_metrics,
        'inference_time': avg_inference_time,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }

def quantize_model(model, calibration_data_loader, device):
    """
    Quantize model to 16-bit precision
    
    Args:
        model: Model to quantize
        calibration_data_loader: Data loader for calibration
        device: Device to use
        
    Returns:
        Quantized model
    """
    # Use PyTorch's quantization tools
    model.eval()
    
    # Convert to float16
    model_fp16 = model.half()
    
    # Run calibration data through the model
    print("Calibrating quantized model...")
    with torch.no_grad():
        for data, _ in calibration_data_loader:
            data = data.to(device).half()
            batch_size = data.shape[0]
            data_flat = data.view(batch_size, -1)
            _ = model_fp16(data_flat)
    
    return model_fp16

def main(args):
    """
    Main function for model compression
    
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
    
    # Create calibration loader with smaller batch size for quantization
    calibration_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=args.batch_size//2, shuffle=True
    )
    
    # Load teacher model
    input_size = data_info['input_size']
    output_size = 4  # SBP, MAP, DBP, HR
    
    teacher_model = MoE(
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
    
    # Load pre-trained weights if provided
    if args.teacher_weights:
        teacher_model.load_state_dict(torch.load(args.teacher_weights))
        print(f"Loaded teacher weights from {args.teacher_weights}")
    else:
        print("Warning: No teacher weights provided. Using randomly initialized weights.")
    
    teacher_model = teacher_model.to(device)
    
    # Evaluate teacher model
    teacher_metrics = evaluate_model_metrics(teacher_model, test_loader, device, "Teacher Model")
    
    # Create student model (pruned version of teacher)
    print(f"\nApplying structured pruning with target sparsity {args.target_sparsity}...")
    student_model = MoE(
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
    student_model.load_state_dict(teacher_model.state_dict())
    student_model = structured_prune_model(student_model, args.target_sparsity)
    student_model = student_model.to(device)
    
    # Evaluate pruned student model
    pruned_metrics = evaluate_model_metrics(student_model, test_loader, device, "Pruned Student Model")
    
    # Apply bidirectional knowledge distillation
    print("\nApplying bidirectional knowledge distillation...")
    for epoch in range(args.distillation_epochs):
        print(f"Distillation Epoch {epoch+1}/{args.distillation_epochs}")
        teacher_model, student_model = bidirectional_knowledge_distillation(
            teacher_model, student_model, train_loader, device
        )
    
    # Evaluate distilled student model
    distilled_metrics = evaluate_model_metrics(student_model, test_loader, device, "Distilled Student Model")
    
    # Apply quantization
    print("\nApplying 16-bit quantization...")
    quantized_model = quantize_model(student_model, calibration_loader, device)
    
    # Evaluate quantized model
    quantized_metrics = evaluate_model_metrics(quantized_model, test_loader, device, "Quantized Model (16-bit)")
    
    # Save models
    torch.save(teacher_model.state_dict(), os.path.join(args.output_dir, "teacher_model.pt"))
    torch.save(student_model.state_dict(), os.path.join(args.output_dir, "student_model.pt"))
    torch.save(quantized_model.state_dict(), os.path.join(args.output_dir, "quantized_model.pt"))
    
    # Print summary
    print("\n--- Compression Summary ---")
    print(f"Original Model Size: {teacher_metrics['model_size_mb']:.2f} MB")
    print(f"Original Parameter Count: {teacher_metrics['parameter_count']:,}")
    print(f"Pruned Model Size: {pruned_metrics['model_size_mb']:.2f} MB")
    print(f"Pruned Parameter Count: {pruned_metrics['parameter_count']:,}")
    print(f"Size Reduction: {(1 - pruned_metrics['model_size_mb']/teacher_metrics['model_size_mb'])*100:.2f}%")
    print(f"Inference Time Improvement: {(1 - pruned_metrics['inference_time']/teacher_metrics['inference_time'])*100:.2f}%")
    
    print("\n--- Model Performance Comparison ---")
    for task in ['SBP', 'MAP', 'DBP', 'HR']:
        print(f"\n{task} Metrics:")
        print(f"  Teacher MAE: {teacher_metrics['task_metrics'][task]['MAE']:.2f}")
        print(f"  Pruned MAE: {pruned_metrics['task_metrics'][task]['MAE']:.2f}")
        print(f"  Distilled MAE: {distilled_metrics['task_metrics'][task]['MAE']:.2f}")
        print(f"  Quantized MAE: {quantized_metrics['task_metrics'][task]['MAE']:.2f}")
    
    print("\nCompression completed successfully!")
    print(f"Models saved in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress LSM²oE model for efficient deployment")
    
    # Data parameters
    parser.add_argument('--ppg_file', type=str, required=True, help='Path to PPG signals file')
    parser.add_argument('--labels_file', type=str, required=True, help='Path to labels file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of remaining data for validation')
    
    # Model parameters
    parser.add_argument('--num_experts', type=int, default=16, help='Number of experts in MoE')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels in expert networks')
    parser.add_argument('--noisy_gating', action='store_true', help='Use noise in gating network')
    parser.add_argument('--noise_type', type=str, default='uniform', help='Type of noise for gating network')
    parser.add_argument('--k', type=int, default=3, help='Number of experts to select for each input')
    parser.add_argument('--expert_type', type=str, default='resnet', help='Type of expert network')
    parser.add_argument('--use_msfgm', action='store_true', help='Use MSFGM in ResNet blocks')
    
    # Compression parameters
    parser.add_argument('--target_sparsity', type=float, default=0.8, help='Target sparsity for pruning')
    parser.add_argument('--distillation_epochs', type=int, default=10, help='Number of distillation epochs')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Miscellaneous
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')
    parser.add_argument('--teacher_weights', type=str, default=None, help='Path to pre-trained teacher weights')
    parser.add_argument('--output_dir', type=str, default='./compressed', help='Output directory')
    
    args = parser.parse_args()
    main(args)