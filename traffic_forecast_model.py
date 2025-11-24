"""
Traffic Volume Prediction from Weather Forecasts

Given two weather forecast texts (A and B), predict which will have more traffic
and by what percentage.

Dataset: Metro Interstate Traffic Volume (Minneapolis-St Paul, MN)
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz
"""

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
BERT_MODEL = "distilbert-base-uncased"
MODEL_PATH = "traffic_model.pkl"
BATCH_SIZE = 100
RIDGE_ALPHA = 1.0
TRAIN_SPLIT = 0.8


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def kelvin_to_fahrenheit(k):
    """Convert Kelvin to Fahrenheit"""
    return (k - 273.15) * 9/5 + 32


def generate_simple_forecast(row):
    """Generate simple forecast text from hourly weather data"""
    temp_f = kelvin_to_fahrenheit(row['temp'])
    rain_text = f"{row['rain_1h']}mm rain" if row['rain_1h'] > 0 else "dry"
    snow_text = f"{row['snow_1h']}mm snow" if row['snow_1h'] > 0 else "no snow"

    return (
        f"Weather is {row['weather_description']}. "
        f"Clouds at {row['clouds_all']}%. Conditions: {rain_text}, {snow_text}. "
        f"Temp is approx {int(temp_f)} degrees."
    )


def generate_daily_forecast(row):
    """Generate daily forecast text from aggregated data"""
    temp_min_f = kelvin_to_fahrenheit(row['temp_min'])
    temp_max_f = kelvin_to_fahrenheit(row['temp_max'])
    rain_text = f"{row['rain_1h_sum']:.2f}mm rain" if row['rain_1h_sum'] > 0 else "dry"
    snow_text = f"{row['snow_1h_sum']:.2f}mm snow" if row['snow_1h_sum'] > 0 else "no snow"

    return (
        f"Daily weather is {row['weather_description_mode']}. "
        f"Clouds average at {row['clouds_all_mean']:.0f}%. "
        f"Conditions: {rain_text}, {snow_text}. "
        f"Min temp approx {int(temp_min_f)} degrees, Max temp approx {int(temp_max_f)} degrees."
    )


def mode_agg(series):
    """Get mode value from series"""
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else np.nan


# =============================================================================
# DATA LOADING & PROCESSING
# =============================================================================

def load_and_aggregate_data(data_path=None):
    """
    Load traffic data and aggregate to daily level.

    Returns:
        daily_df: DataFrame with daily aggregated data
    """
    if data_path is None:
        data_path = DATA_URL

    print("Loading data...")
    df = pd.read_csv(data_path, compression='gzip')
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)
    print(f"Loaded {len(df)} hourly records")

    # Aggregate to daily
    print("Aggregating to daily level...")
    df['date'] = df['date_time'].dt.date

    daily_df = df.groupby('date').agg(
        traffic_volume_sum=('traffic_volume', 'sum'),
        temp_min=('temp', 'min'),
        temp_max=('temp', 'max'),
        weather_description_mode=('weather_description', mode_agg),
        clouds_all_mean=('clouds_all', 'mean'),
        rain_1h_sum=('rain_1h', 'sum'),
        snow_1h_sum=('snow_1h', 'sum'),
        original_entries_count=('date_time', 'count')
    ).reset_index()

    # Generate simple forecast texts
    daily_df['forecast_text'] = daily_df.apply(generate_daily_forecast, axis=1)

    # Calculate normalized traffic volume (average hourly traffic per day)
    daily_df['normalized_traffic_volume'] = (
        daily_df['traffic_volume_sum'] / daily_df['original_entries_count']
    )

    print(f"Aggregated to {len(daily_df)} daily records\n")

    return daily_df


def load_llm_forecasts(daily_df, csv_path='daily_df.csv'):
    """
    Load pre-generated LLM forecasts from CSV if available.

    Args:
        daily_df: DataFrame with daily data
        csv_path: Path to CSV with LLM forecasts

    Returns:
        daily_df with 'llm_forecast_text' column added
    """
    import os
    if os.path.exists(csv_path):
        print(f"Loading pre-generated LLM forecasts from {csv_path}...")
        df_with_llm = pd.read_csv(csv_path)
        if 'llm_forecast_text' in df_with_llm.columns:
            daily_df['llm_forecast_text'] = df_with_llm['llm_forecast_text']
            print(f"Loaded {len(daily_df)} LLM forecasts\n")
        else:
            print(f"Warning: No 'llm_forecast_text' column found in {csv_path}")
    else:
        print(f"No pre-generated LLM forecasts found at {csv_path}")
        print("Using only simple forecasts\n")

    return daily_df


# =============================================================================
# MODEL: BERT EMBEDDINGS + RIDGE REGRESSION
# =============================================================================

def setup_bert():
    """Load BERT model and tokenizer"""
    print("Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL)
    model = DistilBertModel.from_pretrained(BERT_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}\n")
    return tokenizer, model, device


def get_embeddings(text_list, tokenizer, model, device):
    """Convert texts to BERT embeddings"""
    inputs = tokenizer(text_list, padding=True, truncation=True,
                      max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Use CLS token as sentence representation
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embeddings


def generate_embeddings_batched(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    """Generate embeddings for list of texts in batches"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i : i + batch_size]
        batch_emb = get_embeddings(batch, tokenizer, model, device)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)


def train_model(daily_df, tokenizer, model, device):
    """
    Train traffic prediction model.

    Args:
        daily_df: DataFrame with daily data (must have 'forecast_text')
        tokenizer, model, device: BERT components

    Returns:
        regressor: Trained Ridge regression model
        metrics: Dictionary with performance metrics
    """
    print("="*60)
    print("TRAINING MODEL")
    print("="*60 + "\n")

    # Generate embeddings from simple forecasts
    print("Generating embeddings from simple forecast texts...")
    texts_simple = daily_df['forecast_text'].fillna("").tolist()
    X_simple = generate_embeddings_batched(texts_simple, tokenizer, model, device)
    print(f"Simple embeddings shape: {X_simple.shape}\n")

    # Check for LLM forecasts and combine if available
    if 'llm_forecast_text' in daily_df.columns:
        print("Generating embeddings from LLM forecast texts...")
        texts_llm = daily_df['llm_forecast_text'].fillna("").tolist()
        X_llm = generate_embeddings_batched(texts_llm, tokenizer, model, device)
        print(f"LLM embeddings shape: {X_llm.shape}\n")

        # Combine both embeddings
        X = np.concatenate((X_simple, X_llm), axis=1)
        print(f"Combined embeddings shape: {X.shape}\n")
    else:
        X = X_simple

    # Target variable
    y = daily_df['normalized_traffic_volume'].values

    # Split train/test
    split_idx = int(len(daily_df) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training set: {len(X_train)} days")
    print(f"Test set: {len(X_test)} days\n")

    # Train Ridge Regression
    print("Training Ridge Regression...")
    regressor = Ridge(alpha=RIDGE_ALPHA)
    regressor.fit(X_train, y_train)

    # Evaluate
    train_preds = regressor.predict(X_train)
    test_preds = regressor.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"Train RMSE: {train_rmse:.2f} people/hour")
    print(f"Test RMSE:  {test_rmse:.2f} people/hour\n")

    return regressor, {'train_rmse': train_rmse, 'test_rmse': test_rmse}


def save_model(regressor, tokenizer, model, device, model_path=MODEL_PATH):
    """Save trained model"""
    model_data = {
        'regressor': regressor,
        'tokenizer': tokenizer,
        'bert_model': model,
        'device': str(device)
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {model_path}\n")


def load_model(model_path=MODEL_PATH):
    """Load trained model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    regressor = model_data['regressor']
    tokenizer = model_data['tokenizer']
    bert_model = model_data['bert_model']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)

    print(f"Model loaded from {model_path}")
    print(f"Using device: {device}\n")

    return regressor, tokenizer, bert_model, device


# =============================================================================
# PREDICTION & COMPARISON
# =============================================================================

def compare_forecasts(forecast_a, forecast_b, regressor, tokenizer, model, device):
    """
    Compare two weather forecasts and predict traffic difference.

    Args:
        forecast_a: First weather forecast text
        forecast_b: Second weather forecast text
        regressor, tokenizer, model, device: Model components

    Returns:
        winner: 'A' or 'B'
        pct_diff: Percentage difference
        pred_a: Predicted traffic for forecast A
        pred_b: Predicted traffic for forecast B
    """
    # Get embeddings for both forecasts
    emb_a = get_embeddings([forecast_a], tokenizer, model, device)
    emb_b = get_embeddings([forecast_b], tokenizer, model, device)

    # Predict traffic
    pred_a = regressor.predict(emb_a)[0]
    pred_b = regressor.predict(emb_b)[0]

    # Calculate winner and percentage difference
    if pred_a > pred_b:
        winner = 'A'
        pct_diff = ((pred_a - pred_b) / pred_b) * 100
    else:
        winner = 'B'
        pct_diff = ((pred_b - pred_a) / pred_a) * 100

    return winner, pct_diff, pred_a, pred_b


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Main training workflow"""

    print("="*60)
    print("TRAFFIC FORECAST MODEL")
    print("="*60 + "\n")

    # Load and process data
    daily_df = load_and_aggregate_data()

    # Load pre-generated LLM forecasts if available
    daily_df = load_llm_forecasts(daily_df)

    # Setup BERT
    tokenizer, bert_model, device = setup_bert()

    # Train model
    regressor, metrics = train_model(daily_df, tokenizer, bert_model, device)

    # Save model
    save_model(regressor, tokenizer, bert_model, device)

    # Test with example forecasts
    print("="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60 + "\n")

    forecast_a = "Clear skies with temperatures in the lower 60s"
    forecast_b = "Cloudy with a small chance of rain and temperatures in the lower 70s"

    winner, pct_diff, pred_a, pred_b = compare_forecasts(
        forecast_a, forecast_b, regressor, tokenizer, bert_model, device
    )

    print(f"Forecast A: {forecast_a}")
    print(f"Predicted traffic: {pred_a:.2f} people/hour\n")

    print(f"Forecast B: {forecast_b}")
    print(f"Predicted traffic: {pred_b:.2f} people/hour\n")

    print("="*60)
    print(f"RESULT: {winner}, {pct_diff:.2f}%")
    print("="*60)
    print(f"Forecast {winner} is predicted to have {pct_diff:.2f}% more traffic\n")


def predict_mode():
    """Load model and make predictions"""

    print("="*60)
    print("FORECAST COMPARISON MODE")
    print("="*60 + "\n")

    # Load model
    regressor, tokenizer, bert_model, device = load_model()

    # Get user input
    print("Enter two weather forecasts to compare:\n")
    forecast_a = input("Forecast A: ").strip()
    forecast_b = input("Forecast B: ").strip()

    if not forecast_a or not forecast_b:
        print("\nError: Both forecasts must be non-empty.")
        return

    # Compare
    winner, pct_diff, pred_a, pred_b = compare_forecasts(
        forecast_a, forecast_b, regressor, tokenizer, bert_model, device
    )

    print(f"\n{'='*60}")
    print(f"Forecast A traffic: {pred_a:.2f} people/hour")
    print(f"Forecast B traffic: {pred_b:.2f} people/hour\n")
    print(f"RESULT: {winner}, {pct_diff:.2f}%")
    print("="*60)
    print(f"Forecast {winner} is expected to have {pct_diff:.2f}% more traffic")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_mode()
    else:
        main()
