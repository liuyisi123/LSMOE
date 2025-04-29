import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_metrics(y_true, y_pred):
    """
    Evaluate prediction metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Basic error metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    # Mean and standard deviation of errors
    mean_error = np.mean(y_true - y_pred)
    sd_error = np.std(y_true - y_pred)
    
    # Calculate absolute error distribution
    abs_errors = np.abs(y_true - y_pred)
    within_5 = (np.sum(abs_errors <= 5)) / len(abs_errors)
    within_10 = (np.sum(abs_errors <= 10)) / len(abs_errors)
    within_15 = (np.sum(abs_errors <= 15)) / len(abs_errors)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    metrics = {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'MEAN': mean_error,
        'SD': sd_error,
        'MAPE': mape,
        'Within_5': within_5,
        'Within_10': within_10,
        'Within_15': within_15
    }
    
    return metrics

def print_metrics(metrics):
    """
    Print evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
    """
    print('RMSE:', metrics['RMSE'])
    print('R2:', metrics['R2'])
    print('MAE:', metrics['MAE'])
    print('MSE:', metrics['MSE'])
    print('MEAN:', metrics['MEAN'])
    print('SD:', metrics['SD'])
    print('MAPE:', metrics['MAPE'])
    print('Within 5 units:', metrics['Within_5'])
    print('Within 10 units:', metrics['Within_10'])
    print('Within 15 units:', metrics['Within_15'])

def evaluate_bp_standards(metrics):
    """
    Evaluate if BP predictions meet international standards
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary of standard compliance results
    """
    standards = {}
    
    # IEEE Standard
    if metrics['MAE'] <= 5:
        standards['IEEE'] = 'A (Excellent)'
    elif metrics['MAE'] <= 6:
        standards['IEEE'] = 'B (Good)'
    elif metrics['MAE'] <= 7:
        standards['IEEE'] = 'C (Acceptable)'
    else:
        standards['IEEE'] = 'D (Fail)'
    
    # AAMI Standard
    if abs(metrics['MEAN']) <= 5 and metrics['SD'] <= 8:
        standards['AAMI'] = 'Pass'
    else:
        standards['AAMI'] = 'Fail'
    
    # BHS Standard
    if metrics['Within_5'] >= 0.60 and metrics['Within_10'] >= 0.85 and metrics['Within_15'] >= 0.95:
        standards['BHS'] = 'A'
    elif metrics['Within_5'] >= 0.50 and metrics['Within_10'] >= 0.75 and metrics['Within_15'] >= 0.90:
        standards['BHS'] = 'B'
    elif metrics['Within_5'] >= 0.40 and metrics['Within_10'] >= 0.65 and metrics['Within_15'] >= 0.85:
        standards['BHS'] = 'C'
    else:
        standards['BHS'] = 'D'
    
    return standards