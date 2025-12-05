import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Union, Optional

def calculateMetrics(actuals: np.ndarray, predictions: np.ndarray, originalTestTarget: Optional[np.ndarray] = None) -> Dict[str, float]:
    if actuals is None or predictions is None or len(actuals) == 0:
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'nrmse': np.nan}

    actuals = actuals.flatten()
    predictions = predictions.flatten()

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    targetDataForRange = originalTestTarget if originalTestTarget is not None else actuals
    
    try:
        targetRange = np.ptp(targetDataForRange)
        if targetRange == 0 or not np.isfinite(targetRange):
            targetRange = np.ptp(actuals)
    except Exception:
        targetRange = np.ptp(actuals)

    nrmse = rmse / targetRange if targetRange > 1e-9 else np.inf

    return { 
        'mse': float(mse), 
        'rmse': float(rmse), 
        'mae': float(mae), 
        'nrmse': float(nrmse) 
    }