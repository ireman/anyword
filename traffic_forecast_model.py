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
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
BERT_MODEL = "distilbert-base-uncased"
MODEL_PATH = "traffic_model.pkl"
DAILY_DATA_CSV = "daily_df_llm_predictions.csv"
BATCH_SIZE = 100
RIDGE_ALPHA = 1.0
TRAIN_SPLIT = 0.8
MAX_TOKEN_LENGTH = 128

# OpenAI settings for LLM forecast generation
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 150


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def kelvin_to_fahrenheit(k):
    """Convert Kelvin to Fahrenheit"""
    return (k - 273.15) * 9/5 + 32


def describe_temp(f):
    """Generate temperature description"""
    return f"approx {int(f)} degrees"


def generate_hourly_forecast(row):
    """Generate simple forecast text from hourly weather data"""
    temp_f = kelvin_to_fahrenheit(row['temp'])
    rain_text = f"{row['rain_1h']}mm rain" if row['rain_1h'] > 0 else "dry"
    snow_text = f"{row['snow_1h']}mm snow" if row['snow_1h'] > 0 else "no snow"

    return (
        f"Weather is {row['weather_description']}. "
        f"Clouds at {row['clouds_all']}%. Conditions: {rain_text}, {snow_text}. "
        f"Temp is {describe_temp(temp_f)}."
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
        f"Min temp {describe_temp(temp_min_f)}, Max temp {describe_temp(temp_max_f)}."
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
        hourly_df: Original hourly DataFrame (for LLM generation)
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

    # Generate simple daily forecast texts
    daily_df['forecast_text'] = daily_df.apply(generate_daily_forecast, axis=1)

    # Calculate normalized traffic volume (average hourly traffic per day)
    daily_df['normalized_traffic_volume'] = (
        daily_df['traffic_volume_sum'] / daily_df['original_entries_count']
    )

    print(f"Aggregated to {len(daily_df)} daily records\n")

    return daily_df, df


def generate_llm_forecasts(daily_df, hourly_df, api_key):
    """
    Generate detailed LLM-based weather forecasts using OpenAI.
    This matches the notebook implementation in cell 4.

    Args:
        daily_df: DataFrame with daily aggregated data
        hourly_df: DataFrame with hourly data
        api_key: OpenAI API key

    Returns:
        daily_df with 'llm_forecast_text' column added
    """
    import openai

    print("Generating OpenAI LLM daily forecasts (this may take time)...")
    print("This will generate forecasts for all 1,860 days...")

    openai_client = openai.OpenAI(api_key=api_key)
    llm_forecasts = []

    for index, row in tqdm(daily_df.iterrows(), total=len(daily_df)):
        current_date = row['date']

        # Get hourly data for this date
        hourly_data = hourly_df[hourly_df['date'] == current_date].copy()

        # Build hourly narrative
        hourly_narrative = []
        for _, hr_row in hourly_data.iterrows():
            hour = pd.to_datetime(hr_row['date_time']).hour
            hr_temp_f = kelvin_to_fahrenheit(hr_row['temp'])
            hr_clouds = hr_row['clouds_all']
            hr_weather_desc = hr_row['weather_description']
            hr_rain_text = f"{hr_row['rain_1h']}mm rain" if hr_row['rain_1h'] > 0 else "no rain"
            hr_snow_text = f"{hr_row['snow_1h']}mm snow" if hr_row['snow_1h'] > 0 else "no snow"

            hourly_narrative.append(
                f"At {hour:02d}:00, the temperature was {int(hr_temp_f)}°F, with {hr_clouds}% clouds. "
                f"Conditions were '{hr_weather_desc}', and there was {hr_rain_text} and {hr_snow_text}."
            )

        full_hourly_narrative = "\n".join(hourly_narrative)

        # Create prompt
        prompt_message = (
            f"Generate a detailed daily weather forecast transcript. "
            f"Base this forecast on the following hourly weather details:\n\n"
            f"{full_hourly_narrative}\n\n"
            f"Provide a comprehensive narrative forecast that describes weather transitions throughout the day. "
            f"Only output the forecast text, without any conversational elements, leading phrases (e.g., 'Here is your forecast'), "
            f"or concluding remarks. Start directly with the forecast narrative."
        )

        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful weather forecaster who provides concise and detailed weather narratives without conversational filler. Only provide the forecast text."},
                    {"role": "user", "content": prompt_message}
                ],
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=0.0
            )
            generated_text = response.choices[0].message.content
        except Exception as e:
            generated_text = f'Error generating forecast for {current_date}: {e}'

        llm_forecasts.append(generated_text)

    # Add to dataframe
    daily_df['llm_forecast_text'] = llm_forecasts

    # Save to CSV
    daily_df.to_csv(DAILY_DATA_CSV, index=False)
    print(f"\nSaved daily data with LLM forecasts to {DAILY_DATA_CSV}\n")

    return daily_df


def load_or_generate_data():
    """
    Load pre-existing daily data with LLM forecasts, or generate if not found.
    """
    if os.path.exists(DAILY_DATA_CSV):
        print(f"Loading pre-generated data from {DAILY_DATA_CSV}...")
        daily_df = pd.read_csv(DAILY_DATA_CSV)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        print(f"Loaded {len(daily_df)} daily records with LLM forecasts\n")
        return daily_df
    else:
        print(f"{DAILY_DATA_CSV} not found.")
        print("Need to generate LLM forecasts...")

        # Check for OpenAI API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("\nERROR: OPENAI_API_KEY environment variable not set.")
            print("Please set it: export OPENAI_API_KEY='your-key-here'")
            print("\nOr place daily_df_llm_predictions.csv in the current directory.")
            raise ValueError("Missing OPENAI_API_KEY and no pre-generated data found")

        # Load and aggregate data
        daily_df, hourly_df = load_and_aggregate_data()

        # Generate LLM forecasts
        daily_df = generate_llm_forecasts(daily_df, hourly_df, api_key)

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
                      max_length=MAX_TOKEN_LENGTH, return_tensors="pt")
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
    Train traffic prediction model using combined text embeddings.

    Args:
        daily_df: DataFrame with daily data (must have 'forecast_text' and 'llm_forecast_text')
        tokenizer, model, device: BERT components

    Returns:
        regressor: Trained Ridge regression model
        metrics: Dictionary with performance metrics
    """
    print("="*60)
    print("TRAINING MODEL")
    print("="*60 + "\n")

    # Concatenate simple and LLM forecast texts BEFORE embedding
    print("Combining simple and LLM forecast texts...")
    texts_simple = daily_df['forecast_text'].fillna("").tolist()
    texts_llm = daily_df['llm_forecast_text'].fillna("").tolist()

    # Concatenate both texts into single strings
    combined_texts = [f"{simple} {llm}" for simple, llm in zip(texts_simple, texts_llm)]
    print(f"Combined {len(combined_texts)} text pairs\n")

    # Generate embeddings from combined texts
    print("Generating embeddings from combined forecast texts...")
    X = generate_embeddings_batched(combined_texts, tokenizer, model, device)
    print(f"Combined embeddings shape: {X.shape}\n")

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
    # Model was trained on concatenated text, so each input is embedded as-is
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

    # Load or generate daily data with LLM forecasts
    daily_df = load_or_generate_data()

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
