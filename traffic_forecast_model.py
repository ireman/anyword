"""
Traffic Volume Prediction from Weather Forecasts
================================================

This script builds a predictive model that estimates the effect of daily weather
forecasts on traffic volume. The model compares two weather forecast texts and
predicts which will have more traffic and by what percentage.

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
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the model"""
    # Data
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    LOCAL_DATA_PATH = "Metro_Interstate_Traffic_Volume.csv.gz"

    # Model
    BERT_MODEL = "distilbert-base-uncased"
    MAX_TOKEN_LENGTH = 128
    BATCH_SIZE = 100
    RIDGE_ALPHA = 1.0
    TRAIN_SPLIT = 0.8

    # Paths
    MODEL_PATH = "traffic_model.pkl"
    TOKENIZER_PATH = "tokenizer"
    BERT_PATH = "bert_model"

    # OpenAI (optional - for generating LLM forecasts)
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS = 150


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def kelvin_to_fahrenheit(k: float) -> float:
    """Convert Kelvin to Fahrenheit"""
    return (k - 273.15) * 9/5 + 32


def describe_temp(f: float) -> str:
    """Generate temperature description"""
    return f"approx {int(f)} degrees"


def generate_simple_forecast(row: pd.Series) -> str:
    """
    Generate a simple weather forecast text from row data.

    Args:
        row: DataFrame row with weather information

    Returns:
        Simple forecast text string
    """
    temp_f = kelvin_to_fahrenheit(row['temp'])
    rain_text = f"{row['rain_1h']}mm rain" if row['rain_1h'] > 0 else "dry"
    snow_text = f"{row['snow_1h']}mm snow" if row['snow_1h'] > 0 else "no snow"

    return (
        f"Weather is {row['weather_description']}. "
        f"Clouds at {row['clouds_all']}%. Conditions: {rain_text}, {snow_text}. "
        f"Temp is {describe_temp(temp_f)}."
    )


def generate_daily_forecast(row: pd.Series) -> str:
    """
    Generate a daily weather forecast from aggregated data.

    Args:
        row: DataFrame row with aggregated daily weather information

    Returns:
        Daily forecast text string
    """
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


def mode_agg(series: pd.Series):
    """Get mode, handling multiple modes by picking first"""
    modes = series.mode()
    if not modes.empty:
        return modes.iloc[0]
    return np.nan


def load_and_aggregate_data(data_path: str = None) -> pd.DataFrame:
    """
    Load traffic data and aggregate to daily level.

    Args:
        data_path: Path to data file (URL or local path)

    Returns:
        DataFrame with daily aggregated data
    """
    if data_path is None:
        data_path = Config.DATA_URL

    print("Loading data...")
    df = pd.read_csv(data_path, compression='gzip')
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)

    print(f"Loaded {len(df)} hourly records")

    # Generate hourly forecast texts (for potential LLM generation)
    df['forecast_text'] = df.apply(generate_simple_forecast, axis=1)

    # Aggregate to daily level
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

    # Calculate normalized traffic volume (average hourly traffic for the day)
    daily_df['normalized_traffic_volume'] = (
        daily_df['traffic_volume_sum'] / daily_df['original_entries_count']
    )

    print(f"Aggregated to {len(daily_df)} daily records")

    return daily_df, df


def generate_llm_forecasts_openai(daily_df: pd.DataFrame,
                                   hourly_df: pd.DataFrame,
                                   api_key: str,
                                   max_days: int = None) -> List[str]:
    """
    Generate detailed LLM-based weather forecasts using OpenAI.

    Args:
        daily_df: DataFrame with daily aggregated data
        hourly_df: DataFrame with hourly data
        api_key: OpenAI API key
        max_days: Maximum number of days to generate (None for all)

    Returns:
        List of LLM-generated forecast texts
    """
    import openai

    openai_client = openai.OpenAI(api_key=api_key)
    llm_forecasts = []

    days_to_process = len(daily_df) if max_days is None else min(max_days, len(daily_df))

    print(f"Generating LLM forecasts for {days_to_process} days...")

    for index, row in tqdm(daily_df.head(days_to_process).iterrows(), total=days_to_process):
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
                f"At {hour:02d}:00, temperature {int(hr_temp_f)}°F, {hr_clouds}% clouds, "
                f"'{hr_weather_desc}', {hr_rain_text}, {hr_snow_text}."
            )

        full_hourly_narrative = "\n".join(hourly_narrative)

        # Create prompt
        prompt_message = (
            f"Generate a detailed daily weather forecast. "
            f"Base it on these hourly details:\n\n"
            f"{full_hourly_narrative}\n\n"
            f"Provide a comprehensive narrative describing weather transitions throughout the day. "
            f"Only output the forecast text, without conversational elements or leading phrases."
        )

        try:
            response = openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a weather forecaster who provides concise detailed weather narratives without filler. Only provide the forecast text."},
                    {"role": "user", "content": prompt_message}
                ],
                max_tokens=Config.OPENAI_MAX_TOKENS,
                temperature=0.0
            )
            generated_text = response.choices[0].message.content
        except Exception as e:
            generated_text = f'Error generating forecast for {current_date}: {e}'

        llm_forecasts.append(generated_text)

    # Fill remaining with simple forecasts if not all days processed
    if len(llm_forecasts) < len(daily_df):
        remaining = len(daily_df) - len(llm_forecasts)
        llm_forecasts.extend(daily_df['forecast_text'].iloc[-remaining:].tolist())

    return llm_forecasts


# ============================================================================
# MODEL: EMBEDDING GENERATION & TRAINING
# ============================================================================

class TrafficForecastModel:
    """
    Traffic prediction model using BERT embeddings + Ridge Regression.
    """

    def __init__(self, config: Config = None):
        """Initialize the model"""
        self.config = config or Config()
        self.tokenizer = None
        self.bert_model = None
        self.regressor = None
        self.device = None

    def _setup_bert(self):
        """Load and setup BERT model"""
        if self.tokenizer is None:
            print("Loading DistilBERT model...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.BERT_MODEL)
            self.bert_model = DistilBertModel.from_pretrained(self.config.BERT_MODEL)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model.to(self.device)
            print(f"Using device: {self.device}")

    def get_embeddings(self, text_list: List[str]) -> np.ndarray:
        """
        Convert list of texts to BERT embeddings.

        Args:
            text_list: List of text strings

        Returns:
            Array of embeddings (n_texts, 768)
        """
        self._setup_bert()

        # Tokenize
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.config.MAX_TOKEN_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # Use CLS token (first token) as sentence representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embeddings

    def generate_embeddings_batched(self, texts: List[str],
                                   batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts in batches.

        Args:
            texts: List of text strings
            batch_size: Batch size (defaults to config)

        Returns:
            Array of embeddings (n_texts, 768)
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_text = texts[i : i + batch_size]
            batch_emb = self.get_embeddings(batch_text)
            embeddings.append(batch_emb)

        return np.vstack(embeddings)

    def train(self, daily_df: pd.DataFrame, use_llm_forecasts: bool = True):
        """
        Train the traffic prediction model.

        Args:
            daily_df: DataFrame with daily data, must have 'forecast_text' column
            use_llm_forecasts: If True and 'llm_forecast_text' exists, use combined embeddings
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)

        # Generate embeddings from simple forecasts
        print("\nGenerating embeddings from simple forecast texts...")
        texts_simple = daily_df['forecast_text'].fillna("").tolist()
        X_simple = self.generate_embeddings_batched(texts_simple)
        print(f"Simple embeddings shape: {X_simple.shape}")

        # Check if LLM forecasts are available
        if use_llm_forecasts and 'llm_forecast_text' in daily_df.columns:
            print("\nGenerating embeddings from LLM forecast texts...")
            texts_llm = daily_df['llm_forecast_text'].fillna("").tolist()
            X_llm = self.generate_embeddings_batched(texts_llm)
            print(f"LLM embeddings shape: {X_llm.shape}")

            # Combine embeddings
            X = np.concatenate((X_simple, X_llm), axis=1)
            print(f"Combined embeddings shape: {X.shape}")
        else:
            X = X_simple
            print("Using only simple forecast embeddings")

        # Target: normalized traffic volume
        y = daily_df['normalized_traffic_volume'].values

        # Split train/test
        split_idx = int(len(daily_df) * self.config.TRAIN_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\nTraining set: {X_train.shape[0]} days")
        print(f"Test set: {X_test.shape[0]} days")

        # Train Ridge Regression
        print("\nTraining Ridge Regression...")
        self.regressor = Ridge(alpha=self.config.RIDGE_ALPHA)
        self.regressor.fit(X_train, y_train)

        # Evaluate
        train_preds = self.regressor.predict(X_train)
        test_preds = self.regressor.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(f"Train RMSE: {train_rmse:.2f} people/hour")
        print(f"Test RMSE:  {test_rmse:.2f} people/hour")

        # Additional metrics
        train_mae = np.mean(np.abs(y_train - train_preds))
        test_mae = np.mean(np.abs(y_test - test_preds))
        print(f"Train MAE:  {train_mae:.2f} people/hour")
        print(f"Test MAE:   {test_mae:.2f} people/hour")

        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }

    def predict_traffic(self, forecast_text: str) -> float:
        """
        Predict traffic volume for a single forecast text.

        Args:
            forecast_text: Weather forecast text

        Returns:
            Predicted normalized traffic volume (people/hour)
        """
        embeddings = self.get_embeddings([forecast_text])
        prediction = self.regressor.predict(embeddings)[0]
        return prediction

    def compare_forecasts(self, forecast_a: str, forecast_b: str) -> Tuple[str, float]:
        """
        Compare two weather forecasts and predict traffic difference.

        Args:
            forecast_a: First weather forecast text
            forecast_b: Second weather forecast text

        Returns:
            Tuple of (winner, percentage_difference)
            - winner: 'A' or 'B'
            - percentage_difference: Absolute percentage difference
        """
        pred_a = self.predict_traffic(forecast_a)
        pred_b = self.predict_traffic(forecast_b)

        # Calculate percentage difference
        if pred_a > pred_b:
            winner = 'A'
            pct_diff = ((pred_a - pred_b) / pred_b) * 100
        else:
            winner = 'B'
            pct_diff = ((pred_b - pred_a) / pred_a) * 100

        return winner, pct_diff, pred_a, pred_b

    def save_model(self, model_path: str = None):
        """Save the trained model"""
        if model_path is None:
            model_path = self.config.MODEL_PATH

        model_data = {
            'regressor': self.regressor,
            'config': self.config
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to {model_path}")

    def load_model(self, model_path: str = None):
        """Load a trained model"""
        if model_path is None:
            model_path = self.config.MODEL_PATH

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.regressor = model_data['regressor']
        self.config = model_data['config']

        # Setup BERT for inference
        self._setup_bert()

        print(f"Model loaded from {model_path}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main workflow for training and using the model"""

    print("="*60)
    print("TRAFFIC FORECAST MODEL")
    print("="*60)

    # Step 1: Load and process data
    daily_df, hourly_df = load_and_aggregate_data()

    # Step 2: (Optional) Generate LLM forecasts
    # Uncomment and provide API key to generate LLM forecasts
    """
    import os
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key:
        llm_forecasts = generate_llm_forecasts_openai(
            daily_df, hourly_df, openai_api_key, max_days=100
        )
        daily_df['llm_forecast_text'] = llm_forecasts
        daily_df.to_csv('daily_df_with_llm.csv', index=False)
    """

    # Or load pre-generated LLM forecasts
    if os.path.exists('daily_df.csv'):
        print("\nLoading pre-generated LLM forecasts...")
        daily_df_with_llm = pd.read_csv('daily_df.csv')
        daily_df['llm_forecast_text'] = daily_df_with_llm['llm_forecast_text']
        daily_df['normalized_traffic_volume'] = (
            daily_df['traffic_volume_sum'] / daily_df['original_entries_count']
        )

    # Step 3: Train model
    model = TrafficForecastModel()
    metrics = model.train(daily_df, use_llm_forecasts=True)

    # Step 4: Save model
    model.save_model()

    # Step 5: Test with example forecasts
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)

    forecast_a = "Clear skies with temperatures in the lower 60s"
    forecast_b = "Cloudy with a small chance of rain and temperatures in the lower 70s"

    winner, pct_diff, pred_a, pred_b = model.compare_forecasts(forecast_a, forecast_b)

    print(f"\nForecast A: {forecast_a}")
    print(f"Predicted traffic: {pred_a:.2f} people/hour")

    print(f"\nForecast B: {forecast_b}")
    print(f"Predicted traffic: {pred_b:.2f} people/hour")

    print(f"\n{'='*60}")
    print(f"RESULT: {winner}, {pct_diff:.2f}%")
    print(f"{'='*60}")
    print(f"Forecast {winner} is predicted to have {pct_diff:.2f}% more traffic")

    # Additional test cases
    print("\n" + "="*60)
    print("ADDITIONAL TEST CASES")
    print("="*60)

    test_cases = [
        (
            "Sunny and warm, temperatures in the mid 70s with light winds",
            "Heavy snowfall expected with accumulation of 6 inches, temperatures below freezing"
        ),
        (
            "Morning fog clearing to partly cloudy skies, highs near 65",
            "Overcast with periods of heavy rain throughout the day"
        )
    ]

    for i, (fc_a, fc_b) in enumerate(test_cases, 1):
        winner, pct_diff, pred_a, pred_b = model.compare_forecasts(fc_a, fc_b)
        print(f"\nTest Case {i}:")
        print(f"  A: {fc_a}")
        print(f"  B: {fc_b}")
        print(f"  Result: {winner}, {pct_diff:.2f}%")


def load_and_predict():
    """Load a pre-trained model and make predictions"""

    print("Loading pre-trained model...")
    model = TrafficForecastModel()
    model.load_model()

    # Interactive prediction
    print("\n" + "="*60)
    print("FORECAST COMPARISON")
    print("="*60)

    forecast_a = input("\nEnter Forecast A: ")
    forecast_b = input("Enter Forecast B: ")

    winner, pct_diff, pred_a, pred_b = model.compare_forecasts(forecast_a, forecast_b)

    print(f"\n{'='*60}")
    print(f"Forecast A traffic: {pred_a:.2f} people/hour")
    print(f"Forecast B traffic: {pred_b:.2f} people/hour")
    print(f"\nRESULT: {winner}, {pct_diff:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        load_and_predict()
    else:
        main()
