# Predictive Modeling Project with LSTM and CNN

## Project Description

This project is designed for building, training, and evaluating deep learning models using Long Short-Term Memory (LSTM) and one-dimensional Convolutional Neural Networks (CNN) for classification tasks. The program works with artificially generated data, allowing the exploration of the effectiveness of different neural network architectures.

## Software Requirements

### General Requirements

1. **Programming Language:** Python 3.7 or higher.
2. **Libraries:**
    - TensorFlow for building and training models.
    - NumPy for numerical data manipulation.
    - Pandas for data processing and analysis.
    - Scikit-learn for data preprocessing and model evaluation.
    - Seaborn for data and result visualization.

### Functional Requirements

1. **Data Loading and Preprocessing:**
    - Loading data from CSV files.
    - Scaling data (normalization/standardization).
    - Splitting data into training and testing sets.
2. **Model Creation and Training:**
    - Creating an LSTM model for sequential data processing.
    - Creating a CNN model for spatial data processing.
    - Ability to configure model hyperparameters (number of layers, number of neurons, filter size, etc.).
    - Training models with outputting training metrics (loss, accuracy) to the console.
3. **Model Evaluation and Comparison:**
    - Evaluating models on the test set.
    - Outputting key evaluation metrics (accuracy, F1-score, etc.).
4. **Result Visualization:**
    - Plotting training graphs (loss, accuracy).
    - Visualizing prediction results.

### Non-functional Requirements

1. **Performance:**
    - Optimizing data processing and model training to ensure acceptable execution time.
2. **Scalability:**
    - Ability to work with large data volumes through batch processing.
3. **Modularity:**
    - Structuring code into modules for easier maintenance and functionality expansion.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/predictive-model-project.git
    cd predictive-model-project
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # For Windows use `.venv\Scripts\activate`
    ```

## Usage

1. Place your data in the `data/` directory.
2. Run the main script to train models and make predictions:

    ```bash
    python main.py --train --data_path data/data.csv --target_column target --model_save_path models/ --predictions_save_path predictions/ --model_type LSTM --epochs 50 --batch_size 32
    ```

## Command Line Arguments

- `--train`: flag to start model training.
- `--data_path`: path to the data file.
- `--target_column`: name of the target column.
- `--model_save_path`: path to save the trained model.
- `--predictions_save_path`: path to save predictions.
- `--model_type`: type of model to use (LSTM or CNN).
- `--epochs`: number of training epochs.
- `--batch_size`: batch size for training.

## Result Visualization

After training the models and making predictions, you can plot graphs to visualize model performance and the correspondence of predicted values to actual values.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
