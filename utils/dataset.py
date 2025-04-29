import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PPGDataset(Dataset):
    """
    Dataset class for PPG signals and hemodynamic parameters
    """
    def __init__(self, signals, labels, transform=None):
        """
        Initialize PPG dataset
        
        Args:
            signals: PPG signals [num_samples, signal_length]
            labels: Hemodynamic parameters [num_samples, num_params]
            transform: Optional transform to apply to signals
        """
        self.signals = signals
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        """
        Return dataset size
        """
        return len(self.signals)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Args:
            idx: Sample index
            
        Returns:
            Signal and labels for the sample
        """
        signal = self.signals[idx]
        label = self.labels[idx]
        
        # Apply transform if specified
        if self.transform:
            signal = self.transform(signal)
            
        # Convert to tensor
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return signal, label

def preprocess_ppg(signals):
    """
    Preprocess PPG signals
    
    Args:
        signals: Raw PPG signals [num_samples, signal_length]
        
    Returns:
        Preprocessed signals with derivatives
    """
    # Add channel dimension
    ppg_data = signals[:, np.newaxis, :]
    
    # Compute derivatives
    first_derivative = np.gradient(ppg_data, axis=2)
    second_derivative = np.gradient(first_derivative, axis=2)
    
    # Normalize signals
    normalized_signal = (ppg_data - np.min(ppg_data, axis=2, keepdims=True)) / (
        np.max(ppg_data, axis=2, keepdims=True) - np.min(ppg_data, axis=2, keepdims=True))
    normalized_first_derivative = (first_derivative - np.min(first_derivative, axis=2, keepdims=True)) / (
        np.max(first_derivative, axis=2, keepdims=True) - np.min(first_derivative, axis=2, keepdims=True))
    normalized_second_derivative = (second_derivative - np.min(second_derivative, axis=2, keepdims=True)) / (
        np.max(second_derivative, axis=2, keepdims=True) - np.min(second_derivative, axis=2, keepdims=True))
    
    # Concatenate signals
    preprocessed_signals = np.concatenate([
        normalized_signal, normalized_first_derivative, normalized_second_derivative
    ], axis=1)
    
    # Flatten the channel dimension
    preprocessed_signals = preprocessed_signals[:, 0, :]
    
    return preprocessed_signals

def load_dataset(ppg_file, labels_file, test_size=0.2, val_size=0.2, batch_size=64, random_state=42):
    """
    Load and prepare dataset
    
    Args:
        ppg_file: Path to PPG signals file
        labels_file: Path to labels file
        test_size: Proportion of data for testing
        val_size: Proportion of remaining data for validation
        batch_size: Batch size for data loaders
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of data loaders and dataset info
    """
    # Load data
    wavedata_array = np.load(ppg_file)
    label_data = np.load(labels_file)
    
    # Preprocess PPG signals
    preprocessed_signals = preprocess_ppg(wavedata_array)
    
    # Prepare labels
    labels = np.array(label_data[:, 0:4])  # Assuming 4 hemodynamic parameters
    labels = labels[:, np.newaxis]
    
    # Split data
    X_test, X_remaining, y_test, y_remaining = train_test_split(
        preprocessed_signals, labels, test_size=1-test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining, y_remaining, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Create datasets
    train_dataset = PPGDataset(X_train, y_train)
    val_dataset = PPGDataset(X_val, y_val)
    test_dataset = PPGDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_size': preprocessed_signals.shape[1],
        'output_size': labels.shape[1],
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }