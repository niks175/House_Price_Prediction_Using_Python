# House Price Prediction

A machine learning project that predicts house prices using Linear Regression, with a full preprocessing pipeline, cross-validation, and model persistence.

## 📂 Project Structure

```
├── house_price_prediction.py   # Main training/evaluation script
├── house_price.csv             # Dataset (not included — see below)
├── requirements.txt
└── output/                     # Generated after running (model + plot)
```

## 📊 What This Project Does

1. Loads and validates a housing dataset
2. Preprocesses features:
   - Numeric columns → median imputation + scaling
   - Categorical columns → most-frequent imputation + one-hot encoding
3. Splits data into training and test sets
4. Trains a Linear Regression model inside a scikit-learn pipeline (avoids data leakage)
5. Evaluates performance using 5-fold cross-validation and test-set metrics (MAE, RMSE, R²)
6. Saves:
   - The trained model (`house_price_model.joblib`)
   - A scatter plot of actual vs. predicted prices
7. Predicts the price for a sample house as a demo

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, scikit-learn, joblib

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation

```bash
git clone https://github.com/niks175/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Usage

Place your dataset (e.g. `house_price.csv`) in the project folder, then run:

```bash
python house_price_prediction.py --input house_price.csv --target price
```

**Optional arguments:**
| Flag | Description | Default |
|---|---|---|
| `--input` | Path to the CSV dataset | `house_price.csv` |
| `--target` | Name of the target/price column | `price` |
| `--output-dir` | Where to save the model and plot | `output` |
| `--test-size` | Fraction of data used for testing | `0.2` |
| `--show` | Display the plot interactively | off |

## 📈 Output

After running, check the `output/` folder for:
- `house_price_model.joblib` — the trained, reusable model
- `actual_vs_predicted.png` — visualization of prediction accuracy

Example console output includes MAE, RMSE, R² score, and a predicted price for a sample house.

## 📥 Dataset

This project expects a CSV file with house features and a price column (e.g. from Kaggle's house price datasets). Update the `--target` flag if your price column has a different name.

## 🤝 Contributing

Contributions are welcome — feel free to open an issue or submit a pull request.

## 📄 License

Specify your license here (e.g., MIT).
