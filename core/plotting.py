import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional

COLOR_TARGET = "#005180"
COLOR_PRED = "#AE00C5"
COLOR_ERR = '#404040'

def applyStyle():
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'lines.linewidth': 1.5
    })

def predictionAnalysis(predictions: np.ndarray, actuals: np.ndarray, zoomLimit: int = 500):
    if predictions is None or actuals is None or len(predictions) == 0:
        return

    minLen = min(len(predictions), len(actuals))
    predictions = predictions[:minLen].flatten()
    actuals = actuals[:minLen].flatten()
    
    timeSteps = np.arange(len(actuals))
    absoluteError = np.abs(actuals - predictions)
    zoomLimit = min(zoomLimit, len(actuals))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})
    
    axes[0].plot(timeSteps[:zoomLimit], actuals[:zoomLimit], color='black', label='Target Signal', linewidth=0.8, alpha=0.7)
    axes[0].plot(timeSteps[:zoomLimit], predictions[:zoomLimit], color=COLOR_PRED,  linestyle='--', label='Prediction')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.3)

    axes[1].plot(timeSteps[:zoomLimit], absoluteError[:zoomLimit], color=COLOR_ERR, label='|Error|', linewidth=1)
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_xlabel('Time Step')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.3)
    
    plt.show()

def visualizeFeatureMaps(pipeline, inputData: np.ndarray, sampleIndex: int = 0):
    model = pipeline.model
    model.eval()
    device = pipeline.device

    seqLen = pipeline.seqLength
    sample = inputData[sampleIndex : sampleIndex + seqLen]
    
    xTensor = torch.FloatTensor(sample).transpose(0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        featureMaps = model.tcn(xTensor)
        
    activations = featureMaps[0].cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    im = plt.imshow(activations, aspect='auto', cmap='inferno', origin='lower')
    
    plt.colorbar(im, label='Activation Intensity')
    plt.title(f'Internal State')
    plt.ylabel('Channel Index')
    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.show()

def visualizeFLKernels(pipeline):
    model = pipeline.model
    firstLayer = model.tcn.network[0].conv1
    weights = firstLayer.weight.data.cpu().numpy()
    
    numFilters = weights.shape[0]
    kernelSize = weights.shape[2]
    
    cols = 8
    rows = int(np.ceil(numFilters / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2.5 * rows))
    fig.suptitle(f'First Layer (kernelSize: {kernelSize})', y=1.02)
    
    axesFlat = axes.flatten()
    
    for i in range(numFilters):
        ax = axesFlat[i]
        kernelFilter = weights[i, 0, :]
        
        ax.plot(kernelFilter, color='#2980b9', linewidth=2, marker='.', markersize=4)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f'Filter {i}', fontsize=10)
        ax.set_xticks([])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if np.max(np.abs(kernelFilter)) > 0.5:
            ax.set_facecolor('#f0f8ff')

    for j in range(numFilters, len(axesFlat)):
        axesFlat[j].axis('off')
        
    plt.tight_layout()
    plt.show()