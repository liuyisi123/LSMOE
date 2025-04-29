import torch
import torch.nn as nn
import torch.nn.functional as F

class URLoss(nn.Module):
    """
    Uncertainty Regression Loss for multi-task learning
    """
    def __init__(self, task_indices, initial_log_vars=None):
        """
        Initialize URLoss
        
        Args:
            task_indices: Dictionary mapping task names to index ranges in output
            initial_log_vars: Optional initial log variances for each task
        """
        super(URLoss, self).__init__()
        self.task_indices = task_indices
        self.num_tasks = len(task_indices)
        
        # Initialize uncertainty parameters (log(σ²))
        if initial_log_vars is None:
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        else:
            self.log_vars = nn.Parameter(torch.tensor(initial_log_vars))
            
    def forward(self, outputs, targets, lambda_reg=0.0):
        """
        Compute uncertainty-weighted multi-task loss
        
        Args:
            outputs: Predicted values [batch_size, total_outputs]
            targets: Target values [batch_size, total_outputs]
            lambda_reg: Regularization parameter
            
        Returns:
            Total loss and individual task losses
        """
        total_loss = 0.0
        task_losses = {}
        
        for i, (task_name, (start_idx, end_idx)) in enumerate(self.task_indices.items()):
            # Extract predictions and targets for the current task
            task_preds = outputs[:, start_idx:end_idx]
            task_targets = targets[:, start_idx:end_idx]
            
            # Compute squared error
            squared_error = F.mse_loss(task_preds, task_targets)
            
            # Get precision (1/σ²) from log variance
            precision = torch.exp(-self.log_vars[i])
            
            # Compute weighted loss as per Eq.18 in the manuscript
            task_loss = precision * squared_error + 0.5 * self.log_vars[i]
            
            # Add regularization term if specified
            if lambda_reg > 0:
                task_loss += lambda_reg * torch.abs(self.log_vars[i])
            
            # Add to total loss
            total_loss += task_loss
            task_losses[task_name] = squared_error.item()
        
        return total_loss, task_losses