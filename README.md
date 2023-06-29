# Introduction
This is the repository for the paper "Forecasting Marginal Emissions Factors for Demand-Side Management"

# Reproducing Results
* Follow steps in `MLP_MEFs.ipynb`. Update the hyperparameter settings to the ones reported in `FF_models/hours_since_2018_feature/2023_04_01-05_23_36_PM/experiment_settings.txt`, or experiment with new settings.

# Contents

## Data
* `CAISO_Data_2019_2021_NN.csv`: The data used for modeling and evaluation
* `FF_models/hours_since_2018_feature/2023_04_01-05_23_36_PM/CAISO_Data_2019_2021_NN.with_coeff_preds.csv`: a copy of the input data with additional columns for predicted MEFs, predicted MDFs, and train/validation/test-set assignments.

## Code
* `MLP_MEFs.ipynb`: Notebook for training and evaluating a multi-layer perceptron model to predict MEFs and MDFs
* `MLP_d_emission.ipynb`: Notebook for training and evaluating a multi-layer perceptron model to directly predict change in emissions
* `attention_MEFs.ipynb`: Notebook for training and evaluating an attention-based model to predict MEFs and MDFs
* `lstm_MEFs.ipynb`: Notebook for training and evaluating a lstm model to predict MEFs and MDFs
* `sequence_models_utility.py`: Helper script containing methods utilized by the attention and lstm notebooks

## Results
We include results here for our best model only, which was created using the `MLP_MEFs.ipynb` notebook. Results are under `FF_models/`. This includes:
* The scaler that was fit on our training data which can be used to standardize new data before passing it to our model
* Experiment settings
* Evaluation results
* Model .pth file
* Model predictions on the whole 2 years of data along with train/validation/test-set assignments

# Contact info
Nicholas Klein  
nicklein@umich.edu | nmklein@usc.edu | https://www.linkedin.com/in/nic-klein/
