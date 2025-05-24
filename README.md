# S&P 500 Price Prediction with Deep Learning

This project focuses on predicting the S&P 500 stock index prices using Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN) models. It includes functionalities for data acquisition, preprocessing (with wavelet denoising), model training, hyperparameter optimization, and detailed evaluation, including a trading simulator.

## Project Structure

```
.
├── model_eval.ipynb            # Jupyter Notebook for in-depth model evaluation and custom tests
├── visualise_denoise.ipynb     # Jupyter Notebook for visualizing wavelet denoising effects
├── data/
│   ├── getdata.py              # Script to download S&P 500 data using yfinance
│   └── GSPC.csv                # Downloaded S&P 500 historical data
│                    
├── models_saved/
│   ├── 98lstm_model.pt  # Saved weights for the best LSTM model
│   └── noisecnn_model.pt       # Saved weights for the CNN model (denoised variant)
│   
├── reports/                    # Directory for storing evaluation reports, plots, and phase-specific models
│   ├── Phase 1/
│   │   ├── model_saved/        # Models saved during Phase 1 experiments
│   │   ├── notebook_eval/      # Evaluation notebook outputs from Phase 1
│   │   └── report/             # reports from Phase 1
│   └── Phase 2/
│       ├── model_saved/        # Models saved during Phase 2 experiments
│       ├── notebook_eval/      # Evaluation notebooks outputs from Phase 2
│       └── report/             # reports from Phase 2
├── scripts/
│   ├── __init__.py             # Initializes the 'scripts' package
│   ├── config.py               # Configuration file for project parameters
│   ├── data_loader.py          # Loads data from the CSV file
│   ├── HP.py                   # Script for hyperparameter tuning
│   ├── main_pytorch.py         # Main script for training and evaluating models
│   ├── models_pytorch.py       # Defines LSTM and CNN model architectures
│   ├── preprocess.py           # Handles data preprocessing, scaling, and sequence creation
│   └── train_evaluate_pytorch.py # Contains functions for training and evaluating models
│               
└── requirements.txt            # Python package dependencies 

## Setup

### Prerequisites
*   Python 3.x
*   pip (Python package installer)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cxmko/SP500-Price-Prediction
    cd SP500-Price-Prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data

### Fetching Data
The historical S&P 500 data (ticker: `^GSPC`) can be downloaded using the `data/getdata.py` script:
```bash
python data/getdata.py
```
This will download the data and save it as GSPC.csv. The default date range is from '2000-01-01' to '2025-01-01'. You can modify these dates in the script if needed.

### Data Format
The GSPC.csv file contains standard financial time series data, including 'Date', 'Open', 'High', 'Low', 'Close', and 'Volume' columns.

## Configuration
Key parameters for data processing, model architecture, and training are defined in config.py. This includes:
*   File paths (`DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`)
*   Data parameters (`SEQUENCE_LENGTH`, `TARGET_COL`, `FEATURES`, `PREDICTION_HORIZON`)
*   Training parameters (`BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`)
*   Model-specific hyperparameters (e.g., `LSTM_HIDDEN_SIZE`, `CNN_FILTERS`)
*   Saved model names (`LSTM_MODEL_NAME`, `CNN_MODEL_NAME`)

Adjust these parameters as needed before running scripts.

## Usage

### 1. Training and Evaluating Models
The main script for training and/or evaluating models is main_pytorch.py.

**Command-line arguments:**

*   `--model {lstm,cnn,both}`: Specifies the model to train/evaluate. Default: `lstm`.
*   `--visualize_denoising`: If present, enables visualization of the wavelet denoising effect on the target column during preprocessing.
*   `--load_weights`: If **not** present (i.e., by default), pre-trained model weights (if found in models_saved) will be loaded before training. If this flag **is** present, models will be trained from scratch, ignoring any saved weights.
*   `--plot_evaluation`: If **not** present (i.e., by default), evaluation plots comparing predictions to actual values will be generated and saved in the reports directory. If this flag **is** present, plot generation will be skipped.

**Examples:**

*   **Train and evaluate the LSTM model (loading weights if available, generating plots):**
    ```bash
    python scripts/main_pytorch.py --model lstm
    ```
*   **Train the CNN model from scratch and visualize denoising (generating plots):**
    ```bash
    python scripts/main_pytorch.py --model cnn --load_weights --visualize_denoising
    ```
*   **Train both models, load weights, and skip evaluation plotting:**
    ```bash
    python scripts/main_pytorch.py --model both --plot_evaluation
    ```

The script will:
1.  Load data using data_loader.py.
2.  Preprocess data (denoising, scaling, sequence creation) using preprocess.py.
3.  Initialize models defined in models_pytorch.py.
4.  Train and evaluate models using functions from train_evaluate_pytorch.py.
5.  Save trained models to models_saved.
6.  Save evaluation reports and training history plots to reports.

### 2. Hyperparameter Tuning
The script HP.py is used for hyperparameter tuning, currently configured for the LSTM model.

**Command-line arguments:**

*   `--model_type {lstm}`: Specifies the model type to tune. Currently, only `lstm` is supported. (Required)
*   `--epochs_per_trial <int>`: Number of epochs to train each hyperparameter configuration. Default: `10`.
*   `--max_trials <int>`: Maximum number of hyperparameter combinations to try. If not set, all combinations from the search space will be tested. Default: `None`.
*   `--print_every_n_trials <int>`: How often to print the top configurations found so far. Default: `5`.
*   `--num_top_configs <int>`: Number of top configurations to print at each interval and at the end. Default: `3`.

**Example:**

*   **Tune LSTM hyperparameters, running each trial for 15 epochs and testing a maximum of 20 random combinations:**
    ```bash
    python scripts/HP.py --model_type lstm --epochs_per_trial 15 --max_trials 20
    ```

The script defines a search space in its `define_search_space` function and iterates through combinations, evaluating them based on Mean Absolute Error (MAE) on the test set (using denoised targets).

### 3. Model Evaluation Notebook (model_eval.ipynb)
The model_eval.ipynb notebook provides a more interactive and detailed environment for evaluating trained models. It typically includes:
*   Loading pre-trained models from models_saved.
*   Making predictions on test data.
*   Visualizing predictions against actual prices.
*   In-depth error analysis (distribution, residuals, error by price range/volatility).
*   Prediction confidence intervals.
*   Testing predictions on custom sequences.
*   Multi-step ahead forecasting.
*   A trading simulator to backtest model performance with different strategies (e.g., model-based, buy-and-hold).

Open and run this notebook in a Jupyter environment after training your models.

### 4. Denoising Visualization Notebook (visualise_denoise.ipynb)
The visualise_denoise.ipynb notebook is dedicated to visualizing the effect of the wavelet denoising technique applied during preprocessing. It helps in understanding how denoising alters the time series data before it's fed to the models.

## Models

### Architectures
Model architectures are defined in models_pytorch.py:
*   **LSTMModel:** A standard LSTM network.
*   **CNNModel:** A 1D Convolutional Neural Network.

Hyperparameters for these models (e.g., number of layers, hidden units, filter sizes) are primarily configured in config.py.

### Saved Models
*   The primary directory for saving and loading models for main_pytorch.py and model_eval.ipynb is models_saved.
*   The `reports/Phase 1/model_saved/` and `reports/Phase 2/model_saved/` directories may contain models saved during specific experimental phases.

## Output
*   **Reports:** Evaluation metrics, model parameters, and training curves are saved as text files and PNG images in the reports directory. Files are timestamped to avoid overwriting.
*   **Plots:** Training history plots and evaluation plots (if enabled) are saved in reports.
*   **Models:** Trained model weights are saved in models_saved.

