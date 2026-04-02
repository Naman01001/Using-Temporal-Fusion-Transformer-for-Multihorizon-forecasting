# Temporal Fusion Transformer for Multihorizon Forecasting

This repository contains a complete pipeline for multi-horizon time-series forecasting using the **Temporal Fusion Transformer (TFT)**. To handle high-dimensional input features efficiently, the pipeline leverages deep **Autoencoders** for non-linear dimensionality reduction before feeding the processed data into the TFT model.

## 📁 Repository Structure

The workflow is divided into three main Jupyter Notebooks:

### 1. `pca.ipynb` (Dimensionality Reduction & Feature Engineering)
This notebook is responsible for training the Autoencoders used to compress specific groups of features.
* Merges multiple days of historical CSV data into a single dataset.
* Builds and trains dense Autoencoders for different feature groups (e.g., `PB`, `BB`, `VB`).
* Compares the explained variance of the Autoencoder's latent space against standard PCA.
* Saves the trained encoder models as `.keras` files (e.g., `PB_T3.keras`, `VB_T3_3col.keras`) to be used in the data preparation and inference pipelines.

### 2. `Databaker.ipynb` (Batch Data Processing)
A utility notebook designed to prepare the compressed dataset.
* Loads the pre-trained Keras Autoencoder models from the saved directories.
* Iterates over the raw datasets, scales the data using standard deviation and mean, and extracts the lower-dimensional latent representations.
* Saves the highly compressed features to a new directory (e.g., `/content/15_sec_last_PCAcompressed`) for faster downstream model training.

### 3. `main_TFT_pipeline.ipynb` (End-to-End Inference Pipeline)
The main pipeline for running inference using the trained Temporal Fusion Transformer.
* Initializes the environment with `pytorch_forecasting` and `lightning` dependencies.
* Loads the pre-trained TFT model from a checkpoint file (`tft-epoch=03-step=47552.ckpt`).
* Dynamically processes the target dataset by scaling the values and passing the raw features through the saved Autoencoders to generate latent features on the fly.
* Handles padding and missing value indicators by creating boolean `_not_nan` indicator columns.
* Formats the final dataframe with a `time_idx` sequence and a `group_id` required by the `TimeSeriesDataSet` for TFT inference.

## 🧠 Methodology

1. **Feature Compression**: High-dimensional indicator groups (such as the 18 `PB` features) are compressed into smaller latent dimensions (e.g., 4 dimensions) using a 3-layer deep Autoencoder. This retains complex non-linear relationships better than standard PCA.
2. **Data Formatting**: The latent representations are concatenated alongside the target (derived from `close.diff()`), aligned using top zero-padding, and enriched with `_not_nan` indicator columns to help the sequence models handle missing data. 
3. **Forecasting**: The formatted, time-indexed data is natively structured so it can be fed to the Temporal Fusion Transformer for multi-horizon prediction.

## 🛠️ Requirements

To run the notebooks, ensure you have the following key dependencies installed:
* `pytorch_forecasting`
* `pytorch-lightning`
* `tensorflow` / `keras`
* `scikit-learn`
* `pandas` and `numpy`
* `tqdm`

## 🚀 How to Run

1. **Train Feature Extractors**: Run `pca.ipynb` to train the Autoencoders on your historical data and save the `.keras` models.
2. **Prepare Training Data**: Use `Databaker.ipynb` to bulk-process and compress your historical datasets using the saved models.
3. **Run Inference**: Open `main_TFT_pipeline.ipynb` to process new CSV data, format the TimeSeriesDataSet properties, and run predictions using the saved TFT checkpoint.
