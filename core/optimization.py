# TCN/core/optimization.py
import optuna
import torch
import numpy as np
from typing import Dict, Any

from .TCNmodel import TCNPipeline
from .metrics import calculateMetrics 

def objective(trial: optuna.Trial, data: np.ndarray, baseConfig: Dict[str, Any]) -> float:
    kernelSize = trial.suggest_categorical('kernelSize', [3, 5, 7])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learningRate = trial.suggest_float('learningRate', 1e-4, 1e-2, log=True)

    numLayers = trial.suggest_int('numLayers', 2, 5)
    numFilters = trial.suggest_categorical('numFilters', [16, 32, 64])
    channels = [numFilters] * numLayers
    

    pipeline = TCNPipeline(
        inputSize=1,
        outputSize=1,
        numChannels=channels,
        kernelSize=kernelSize,
        dropout=dropout,
        learningRate=learningRate,
        seqLength=baseConfig['seqLength'],
        batchSize=baseConfig['batchSize'],
        device=baseConfig['device']
    )
    

    try:
        results = pipeline.run(
            timeseries=data,
            trainRatio=baseConfig['trainRatio'],
            epochs=20,
            patience=5
        )
        
        targets = results['targets']
        predictions = results['predictions']
        metrics_dict = calculateMetrics(targets, predictions)
        metric = metrics_dict['rmse']
        
    except Exception as e:
        print(f"Trial failed with params {trial.params}: {e}")
        metric = float('inf')
        
    return metric

def runHyperparameterTuning(data: np.ndarray, config: Dict[str, Any], nTrials: int = 20) -> Dict[str, Any]:
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data, config), n_trials=nTrials)    
    if study.best_value == float('inf'):
        return config 
        
    print(f"Best RMSE found: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    return study.best_params