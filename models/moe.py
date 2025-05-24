import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from models.resnet import ResNet
from models.msfgm import MSFGM
from models.attention import HybridAttentionModule

class SparseDispatcher(object):
    """
    Handles the routing of inputs to experts
    """
    def __init__(self, num_experts, gates):
        """
        Initialize SparseDispatcher
        
        Args:
            num_experts: Number of experts
            gates: Expert selection gates [batch_size, num_experts]
        """
        self._gates = gates
        self._num_experts = num_experts
        
        # Sort experts by gate values
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        if sorted_experts.numel() == 0:
            # Handle case where no gates are selected
            self._expert_index = torch.empty(0, 1, dtype=torch.long, device=gates.device)
            self._batch_index = torch.empty(0, dtype=torch.long, device=gates.device)
            self._part_sizes = [0] * num_experts
            self._nonzero_gates = torch.empty(0, 1, device=gates.device)
        else:
            _, self._expert_index = sorted_experts.split(1, dim=1)
            self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
            self._part_sizes = (gates > 0).sum(0).tolist()
            
            # Get nonzero gate values
            gates_exp = gates[self._batch_index.flatten()]
            self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        
    def dispatch(self, inp):
        """
        Dispatch input tensor to selected experts
        
        Args:
            inp: Input tensor [batch_size, features]
            
        Returns:
            List of tensors, one per expert
        """
        if self._batch_index.numel() == 0:
            # Return empty tensors for all experts
            return [torch.empty(0, inp.size(1), device=inp.device) for _ in range(self._num_experts)]
        
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        Combine expert outputs according to gates
        
        Args:
            expert_out: List of tensors from experts
            multiply_by_gates: Whether to weight outputs by gate values
            
        Returns:
            Combined output tensor
        """
        # Filter out empty expert outputs
        non_empty_outputs = [out for out in expert_out if out.numel() > 0]
        
        if not non_empty_outputs:
            # If all outputs are empty, return zeros
            output_size = expert_out[0].size(1) if expert_out else 1
            return torch.zeros(self._gates.size(0), output_size, 
                             requires_grad=True, device=self._gates.device)
        
        stitched = torch.cat(non_empty_outputs, 0)
        if multiply_by_gates and self._nonzero_gates.numel() > 0:
            stitched = stitched.mul(self._nonzero_gates)
        
        # Create output tensor of zeros and add expert outputs at appropriate indices
        zeros = torch.zeros(self._gates.size(0), stitched.size(1), 
                           requires_grad=True, device=stitched.device)
        
        if self._batch_index.numel() > 0:
            combined = zeros.index_add(0, self._batch_index, stitched.float())
        else:
            combined = zeros
            
        return combined
    
    def expert_to_gates(self):
        """
        Return gates for each expert
        
        Returns:
            List of gate values for each expert
        """
        if self._nonzero_gates.numel() == 0:
            return [torch.empty(0, 1, device=self._gates.device) for _ in range(self._num_experts)]
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MoE(nn.Module):
    """
    Sparse Multi-gate Mixture-of-Experts model for multi-task learning
    """
    def __init__(self, input_size, output_size, num_experts=16, hidden_channels=128, 
                 noisy_gating=True, noise_type='uniform', k=3, expert_type='resnet',
                 use_msfgm=True):
        """
        Initialize MoE model
        
        Args:
            input_size: Dimension of input features
            output_size: Dimension of output predictions
            num_experts: Number of expert networks
            hidden_channels: Hidden channels in expert networks
            noisy_gating: Whether to use noise in gating network
            noise_type: Type of noise to use ('gaussian', 'uniform', etc.)
            k: Number of experts to select for each input
            expert_type: Type of expert network ('resnet' or 'msfgm')
            use_msfgm: Whether to use MSFGM in each expert
        """
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.noise_type = noise_type
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.k = k
        
        # Create expert networks
        if expert_type == 'resnet':
            self.experts = nn.ModuleList([
                ResNet(1, output_size, hidden_channels, use_msfgm=use_msfgm) 
                for _ in range(num_experts)
            ])
        elif expert_type == 'msfgm':
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(inplace=True),
                    MSFGM(hidden_channels),
                    MSFGM(hidden_channels),
                    MSFGM(hidden_channels),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(hidden_channels, output_size)
                ) for _ in range(num_experts)
            ])
        
        # Gating network parameters
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        
        # Activation functions
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        
        # Register buffers for normal distribution
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        # Ensure k <= num_experts
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """
        Compute coefficient of variation squared
        
        Args:
            x: Input tensor
            
        Returns:
            CV² value (variance / mean²)
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """
        Compute load per expert from gates
        
        Args:
            gates: Gate values [batch_size, num_experts]
            
        Returns:
            Load per expert (number of inputs assigned to each expert)
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        Compute probability of each expert being in top-k
        
        Args:
            clean_values: Clean logits [batch_size, num_experts]
            noisy_values: Noisy logits [batch_size, num_experts]
            noise_stddev: Standard deviation of noise [batch_size, num_experts]
            noisy_top_values: Top-k+1 noisy values [batch_size, k+1]
            
        Returns:
            Probability of each expert being in top-k
        """
        # Compute indices and thresholds
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        
        # Threshold for being in/out of top-k
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        # Compute probabilities using normal CDF
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        
        return prob

    def add_noise(self, clean_logits, noise_stddev):
        """
        Add noise to gate logits
        
        Args:
            clean_logits: Clean gate logits [batch_size, num_experts]
            noise_stddev: Standard deviation of noise [batch_size, num_experts]
            
        Returns:
            Noisy logits
        """
        if self.noise_type == 'gaussian':
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        elif self.noise_type == 'poisson':
            noisy_logits = clean_logits + (torch.poisson(clean_logits * noise_stddev) - clean_logits * noise_stddev)
        elif self.noise_type == 'uniform':
            uniform_noise = torch.rand_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + uniform_noise
        elif self.noise_type == 'beta':
            alpha, beta = 0.5, 0.5
            beta_noise = torch.distributions.Beta(alpha, beta).sample(clean_logits.shape).to(clean_logits.device)
            noisy_logits = clean_logits + beta_noise * noise_stddev
        elif self.noise_type == 'salt_and_pepper':
            mask = torch.rand_like(clean_logits)
            salt = mask > 0.95
            pepper = mask < 0.05
            noisy_logits = clean_logits.clone()
            noisy_logits[salt] = 1.0
            noisy_logits[pepper] = 0.0
        elif self.noise_type == 'speckle':
            speckle_noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + clean_logits * speckle_noise        
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
            
        return noisy_logits

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        Compute gate values with noisy top-k selection
        
        Args:
            x: Input tensor [batch_size, input_size]
            train: Whether in training mode
            noise_epsilon: Minimum noise level
            
        Returns:
            Sparse gate values and load per expert
        """
        # Compute gate logits
        clean_logits = x @ self.w_gate
        
        # Add noise during training
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = self.add_noise(clean_logits, noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            
        # Apply softmax
        logits = self.softmax(logits)
        
        # Select top-k experts
        k = min(self.k + 1, self.num_experts)
        top_logits, top_indices = logits.topk(k, dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        
        # Normalize top-k gate values
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        
        # Create sparse gate tensor
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        # Compute load
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
            
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """
        Forward pass through MoE model
        
        Args:
            x: Input tensor [batch_size, input_size]
            loss_coef: Coefficient for load balancing loss
            
        Returns:
            Model output and load balancing loss
        """
        # Compute gates and load
        gates, load = self.noisy_top_k_gating(x, self.training)
        
        # Compute importance and load balancing loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        
        # Dispatch inputs to experts
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        
        # Add channel dimension for 1D convolution
        expert_inputs = [inp.unsqueeze(1) for inp in expert_inputs]
        
        # Get expert outputs - handle empty inputs
        expert_outputs = []
        for i in range(self.num_experts):
            if expert_inputs[i].numel() == 0:
                # Create empty output with correct shape
                empty_output = torch.zeros(0, self.output_size, 
                                         device=x.device, requires_grad=True)
                expert_outputs.append(empty_output)
            else:
                expert_outputs.append(self.experts[i](expert_inputs[i]))
        
        # Combine expert outputs
        y = dispatcher.combine(expert_outputs)
        
        return y, loss
