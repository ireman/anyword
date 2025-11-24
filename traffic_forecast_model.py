"""
Traffic Volume Prediction from Weather Forecasts
Given two weather forecasts (A and B), predict which has more traffic and by what %.
Dataset: Metro Interstate Traffic Volume (UCI ML Repository)
"""

import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Configuration
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
BERT_MODEL = "distilbert-base-uncased"
MODEL_PATH = "traffic_model.pkl"
DAILY_DATA_CSV = "daily_df_llm_predictions.csv"
BATCH_SIZE = 100
RIDGE_ALPHA = 1.0
TRAIN_SPLIT = 0.7      # 70% for training
VAL_SPLIT = 0.15       # 15% for validation
TEST_SPLIT = 0.15      # 15% for testing
MAX_TOKEN_LENGTH = 128

# OpenAI settings for LLM forecast generation
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 150

# Visualization settings
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Helper Functions
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

# Data Loading & Processing
def load_and_aggregate_data(data_path=None):
    """Load traffic data and aggregate to daily level"""
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
    """Generate LLM-based weather forecasts using OpenAI"""
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
    """Load pre-existing daily data with LLM forecasts, or generate if not found"""
    if os.path.exists(DAILY_DATA_CSV):
        print(f"Loading pre-generated data from {DAILY_DATA_CSV}...")
        daily_df = pd.read_csv(DAILY_DATA_CSV)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        print(f"Loaded {len(daily_df)} daily records with LLM forecasts\n")
        return daily_df
    else:
        print(f"{DAILY_DATA_CSV} not found. Need to generate LLM forecasts...")
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Export it or provide daily_df_llm_predictions.csv")
            raise ValueError("Missing OPENAI_API_KEY and no pre-generated data")

        # Load and aggregate data
        daily_df, hourly_df = load_and_aggregate_data()

        # Generate LLM forecasts
        daily_df = generate_llm_forecasts(daily_df, hourly_df, api_key)

        return daily_df

# BERT Embeddings + Ridge Regression
def setup_bert():
    """Load BERT model and tokenizer with CUDA optimization"""
    print("Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL)
    model = DistilBertModel.from_pretrained(BERT_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model.to(device).eval()
    print()
    return tokenizer, model, device


def get_embeddings(text_list, tokenizer, model, device):
    """Convert texts to BERT embeddings"""
    inputs = tokenizer(text_list, padding=True, truncation=True,
                      max_length=MAX_TOKEN_LENGTH, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return cls_embeddings


def generate_embeddings_batched(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    """Generate embeddings for list of texts in batches"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i : i + batch_size]
        batch_emb = get_embeddings(batch, tokenizer, model, device)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

# Visualization Functions
def plot_training_curves(metrics, save_path=None):
    """Plot train/val/test RMSE to check for overfitting"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Bar chart of RMSE
    datasets = ['Train', 'Validation', 'Test']
    rmse_values = [metrics['train_rmse'], metrics['val_rmse'], metrics['test_rmse']]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    ax1.bar(datasets, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('RMSE (people/hour)', fontsize=12)
    ax1.set_title('Model Performance Across Datasets', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(rmse_values):
        ax1.text(i, v + 10, f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Overfitting analysis
    train_val_diff = metrics['val_rmse'] - metrics['train_rmse']
    val_test_diff = metrics['test_rmse'] - metrics['val_rmse']

    categories = ['Train vs Val\nGap', 'Val vs Test\nGap']
    gaps = [train_val_diff, val_test_diff]
    gap_colors = ['#e67e22' if g > 50 else '#27ae60' for g in gaps]

    ax2.bar(categories, gaps, color=gap_colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('RMSE Difference (people/hour)', fontsize=12)
    ax2.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(gaps):
        ax2.text(i, v + 3, f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add interpretation text
    if train_val_diff < 50:
        overfitting_status = "✓ No significant overfitting"
        status_color = 'green'
    else:
        overfitting_status = "⚠ Potential overfitting detected"
        status_color = 'orange'

    fig.text(0.5, 0.02, overfitting_status, ha='center', fontsize=12,
             color=status_color, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_predictions(y_true, y_pred, dataset_name='Test', save_path=None):
    """Plot actual vs predicted values and residuals"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Actual vs Predicted
    ax1.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Traffic (people/hour)', fontsize=12)
    ax1.set_ylabel('Predicted Traffic (people/hour)', fontsize=12)
    ax1.set_title(f'{dataset_name} Set: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.1f}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Residuals
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Traffic (people/hour)', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.set_title(f'{dataset_name} Set: Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    # Add residual statistics
    residual_std = np.std(residuals)
    ax2.text(0.05, 0.95, f'Residual Std: {residual_std:.1f}',
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")

    plt.show()


def plot_alpha_comparison(alphas, train_rmses, val_rmses, save_path=None):
    """Plot RMSE vs Ridge alpha"""
    plt.figure(figsize=(10, 6))

    plt.plot(alphas, train_rmses, marker='o', linewidth=2, markersize=8,
             label='Training RMSE', color='#2ecc71')
    plt.plot(alphas, val_rmses, marker='s', linewidth=2, markersize=8,
             label='Validation RMSE', color='#3498db')

    plt.xscale('log')
    plt.xlabel('Ridge Alpha (Regularization Strength)', fontsize=12)
    plt.ylabel('RMSE (people/hour)', fontsize=12)
    plt.title('Ridge Regularization: Training vs Validation RMSE', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)

    # Mark the best alpha
    best_alpha_idx = np.argmin(val_rmses)
    best_alpha = alphas[best_alpha_idx]
    plt.axvline(x=best_alpha, color='red', linestyle='--', linewidth=2,
                label=f'Best α = {best_alpha:.2f}')
    plt.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Alpha comparison plot saved to {save_path}")

    plt.show()


def train_model(daily_df, tokenizer, model, device):
    """Train traffic prediction model with validation and overfitting detection"""
    print("="*60)
    print("TRAINING MODEL WITH VALIDATION")
    print("="*60 + "\n")

    print("Combining simple and LLM forecast texts...")
    texts_simple = daily_df['forecast_text'].fillna("").tolist()
    texts_llm = daily_df['llm_forecast_text'].fillna("").tolist()
    combined_texts = [f"{simple} {llm}" for simple, llm in zip(texts_simple, texts_llm)]
    print(f"Combined {len(combined_texts)} text pairs\n")

    print("Generating embeddings...")
    X = generate_embeddings_batched(combined_texts, tokenizer, model, device)
    print(f"Embeddings shape: {X.shape}\n")

    y = daily_df['normalized_traffic_volume'].values

    n = len(daily_df)
    train_idx = int(n * TRAIN_SPLIT)
    val_idx = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    X_train = X[:train_idx]
    y_train = y[:train_idx]

    X_val = X[train_idx:val_idx]
    y_val = y[train_idx:val_idx]

    X_test = X[val_idx:]
    y_test = y[val_idx:]

    print(f"Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test\n")

    print("Finding best Ridge alpha...")
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    train_rmses, val_rmses = [], []

    for alpha in alphas:
        reg = Ridge(alpha=alpha).fit(X_train, y_train)
        train_rmses.append(np.sqrt(mean_squared_error(y_train, reg.predict(X_train))))
        val_rmses.append(np.sqrt(mean_squared_error(y_val, reg.predict(X_val))))

    valid_indices = [i for i, (t, v) in enumerate(zip(train_rmses, val_rmses)) if v >= t]
    best_idx = min(valid_indices, key=lambda i: val_rmses[i]) if valid_indices else np.argmin(val_rmses)
    best_alpha = alphas[best_idx]
    print(f"Best alpha: {best_alpha} (Train: {train_rmses[best_idx]:.2f}, Val: {val_rmses[best_idx]:.2f})\n")

    plot_alpha_comparison(alphas, train_rmses, val_rmses, save_path=f"{PLOT_DIR}/alpha_comparison.png")

    print(f"Training with alpha={best_alpha}...")
    regressor = Ridge(alpha=best_alpha).fit(X_train, y_train)

    train_preds = regressor.predict(X_train)
    val_preds = regressor.predict(X_val)
    test_preds = regressor.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    test_r2 = r2_score(y_test, test_preds)

    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"Train RMSE: {train_rmse:.2f} people/hour  |  R² = {train_r2:.3f}")
    print(f"Val   RMSE: {val_rmse:.2f} people/hour  |  R² = {val_r2:.3f}")
    print(f"Test  RMSE: {test_rmse:.2f} people/hour  |  R² = {test_r2:.3f}\n")

    train_val_gap = val_rmse - train_rmse
    val_test_gap = test_rmse - val_rmse

    print("Overfitting Analysis:")
    print(f"  Train-Val gap: {train_val_gap:+.2f} {'✓' if abs(train_val_gap) < 50 else '⚠'}")
    print(f"  Val-Test gap:  {val_test_gap:+.2f} {'✓' if abs(val_test_gap) < 50 else '⚠'}\n")

    metrics = {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'best_alpha': best_alpha
    }

    print("Generating visualizations...")
    plot_training_curves(metrics, save_path=f"{PLOT_DIR}/training_curves.png")
    plot_predictions(y_test, test_preds, 'Test', save_path=f"{PLOT_DIR}/test_predictions.png")

    print(f"\nAll plots saved to '{PLOT_DIR}/' directory\n")

    return regressor, metrics

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

    bert_model.to(device).eval()

    print(f"Model loaded from {model_path}")
    print(f"Using: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    return regressor, tokenizer, bert_model, device

# Prediction & Comparison
def compare_forecasts(forecast_a, forecast_b, regressor, tokenizer, model, device):
    """Compare two forecasts and predict traffic difference"""
    emb_a = get_embeddings([forecast_a], tokenizer, model, device)
    emb_b = get_embeddings([forecast_b], tokenizer, model, device)

    pred_a = regressor.predict(emb_a)[0]
    pred_b = regressor.predict(emb_b)[0]

    if pred_a > pred_b:
        winner = 'A'
        pct_diff = ((pred_a - pred_b) / pred_b) * 100
    else:
        winner = 'B'
        pct_diff = ((pred_b - pred_a) / pred_a) * 100

    return winner, pct_diff, pred_a, pred_b

# Main Workflow
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
