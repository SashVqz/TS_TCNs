# SolarCycle and TCN

## Project Context
This is a personal research project designed to understand the inner workings of **Temporal Convolutional Networks (TCN)**. The goal is to move beyond standard Recurrent Neural Networks (RNNs) and explore how 1D dilated convolutions can effectively model long-range dependencies in physical phenomena, specifically the 11-year solar Schwabe cycle.

## Architecture & Mathematics

The core model is a **Temporal Convolutional Network** (TCN), a variation of 1D CNNs tailored for sequence modeling. Unlike LSTM/GRU, which process time sequentially, this TCN operates in parallel using a hierarchy of dilated convolutions to build a massive effective history.

The mathematical backbone relies on three specific mechanisms implemented in `TCNmodel.py`:

### 1. Causal Convolutions
To prevent future data leakage (predicting $t$ using $t+1$), the network uses causal padding.
In standard CNNs, a filter sees input $[t-k, t+k]$. Here, the output at time $t$ is convolved only with elements from time $t$ and earlier in the previous layer.
* **Implementation:** In PyTorch, this is achieved by adding padding of length $(k-1) \cdot d$ and then "chomping" (trimming) the end of the output to maintain sequence length.

### 2. Dilated Convolutions (The 11-Year Cycle Solver)
A standard convolution has a linear receptive field growth. To capture the solar cycle (~132 months) without vanishing gradients, we use dilation. The filter $f$ is applied over an input sequence $\mathbf{x}$ with a dilation factor $d$:

$$F(s) = (x *_d f)(s) = \sum_{i=0}^{k-1} f(i) \cdot \mathbf{x}_{s - d \cdot i}$$

* **$d$ (Dilation):** The space between kernel points. In this architecture, $d$ increases exponentially with the depth of the network ($d = 2^i$ at layer $i$).
* **Receptive Field:** This allows the network to look back exponentially far ($2^L$) with a logarithmic number of layers, effectively capturing the 11-year periodicity.

### 3. Residual Blocks
To effectively train deep stacks of these layers, the model uses Residual Connections based on the ResNet architecture:

$$o = \text{Activation}(\mathbf{x} + \mathcal{F}(\mathbf{x}))$$

Where $\mathcal{F}$ is the series of transformations: `Dilated Conv1d` $\to$ `WeightNorm` $\to$ `ReLU` $\to$ `Dropout`. This stabilization is critical for regression on volatile astrophysical data.

## Dataset & Methodology

### Data Source
**Dataset:** Monthly Mean Total Sunspot Number (1749–Present).
**Source:** SIDC - Solar Influences Data Analysis Center.

The dataset represents the **Schwabe Cycle**, a magnetic phenomenon driven by the sun's dynamo mechanism (differential rotation and convective flows).

### Preprocessing Strategy
Raw physical data is rarely ready for Deep Learning. The pipeline (`loadAndPreprocessData`) applies:

1.  **Linear Interpolation:** Physical time series must be continuous. Missing observational months are interpolated to preserve the phase continuity of the magnetic wave.
2.  **Standardization:** Sunspot numbers vary wildly (0 to 400+). A `StandardScaler` ($\mu=0, \sigma=1$) is applied to normalize activations and gradients.
3.  **Sliding Window (Sequence Formulation):**
    * **Input Window:** 144 months (12 years). This strictly forces the model to see at least one full solar cycle before making a prediction.
    * **Horizon:** 1 month (One-step-ahead prediction).
    * **Tensor Shape:** $(N, 1, 144)$.

## Structure

The project is engineered as a modular python package:

* **`TCN/core/`**:
    * `TCNmodel.py`: The PyTorch architecture (TemporalBlock, TCNMain).
    * `optimization.py`: Bayesian hyperparameter search using **Optuna**.
    * `metrics.py`: Regression metrics (RMSE, NRMSE, MAE).
    * `plotting.py`: Inspection tools for internal Feature Maps and Kernels.
* **`notebooks/summary.ipynb`**: Main orchestrator for training and analysis.
* **`data/`**: Contains the CSV dataset.

## Setup & Usage

### 1. Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Bibliography

## Deep Learning

* **Bai, S., Kolter, J. Z., & Koltun, V. (2018).** *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.
    > The foundational paper establishing TCNs as a superior alternative to RNNs/LSTMs for sequence modeling.

## Astrophysical Context

* **Hathaway, D. H. (2015).** The Solar Cycle. *Living Reviews in Solar Physics, 12(1).*
    > Definitive review of the observational characteristics of the 11-year sunspot cycle.

* **Charbonneau, P. (2020).** Dynamo models of the solar cycle. *Living Reviews in Solar Physics.*
    > Explains the magnetohydrodynamic (MHD) forces that generate the time series patterns this model attempts to learn.