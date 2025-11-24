# Quick Start Guide

Get up and running with the Traffic Forecast Model in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- ~2GB disk space for models
- GPU recommended but not required (CPU works, just slower)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data processing)
- torch, transformers (BERT model)
- scikit-learn (regression)
- tqdm (progress bars)

## Training the Model

### Option 1: Use Pre-generated Data (Recommended - Fastest)

If you have the `daily_df.csv` file with pre-generated LLM forecasts:

```bash
python traffic_forecast_model.py
```

This takes **~10-15 minutes** and will:
1. Load the data (1,860 days)
2. Generate BERT embeddings
3. Train the model
4. Save to `traffic_model.pkl`
5. Show example predictions

### Option 2: Generate Everything from Scratch

If you want to generate LLM forecasts (requires OpenAI API key):

1. Edit `traffic_forecast_model.py` and uncomment the OpenAI section:

```python
# Around line 480 in main() function
import os
openai_api_key = os.environ.get('OPENAI_API_KEY')
if openai_api_key:
    llm_forecasts = generate_llm_forecasts_openai(
        daily_df, hourly_df, openai_api_key, max_days=100  # Or None for all days
    )
    daily_df['llm_forecast_text'] = llm_forecasts
    daily_df.to_csv('daily_df_with_llm.csv', index=False)
```

2. Set your API key and run:

```bash
export OPENAI_API_KEY='your-key-here'
python traffic_forecast_model.py
```

**Note:** Generating all 1,860 forecasts costs ~$5-10 in API credits and takes ~1 hour.

## Making Predictions

### Quick Test with Examples

```bash
python example_usage.py
```

Output:
```
Test Case 1: Original Task Example
Forecast A: Clear skies with temperatures in the lower 60s
Forecast B: Cloudy with a small chance of rain and temperatures in the lower 70s

Predicted Traffic:
  Forecast A: 3245.67 people/hour
  Forecast B: 3024.12 people/hour

RESULT: A, 7.32%
Forecast A is expected to have 7.32% more traffic
```

### Interactive Mode

```bash
python example_usage.py interactive
```

Then enter your own forecasts:
```
Forecast A: Sunny and 75 degrees
Forecast B: Heavy rain all day

RESULT: A, 15.43%
```

### Single Forecast Prediction

```bash
python example_usage.py single
```

### Python API

```python
from traffic_forecast_model import TrafficForecastModel

# Load model
model = TrafficForecastModel()
model.load_model('traffic_model.pkl')

# Compare forecasts
forecast_a = "Clear skies with temperatures in the lower 60s"
forecast_b = "Cloudy with rain and temperatures in the lower 70s"

winner, pct_diff, pred_a, pred_b = model.compare_forecasts(forecast_a, forecast_b)

print(f"Result: {winner}, {pct_diff:.2f}%")
# Output: Result: A, 7.32%
```

## Expected Performance

- **Test RMSE:** ~610-615 people/hour
- **Relative Error:** ~17-18%
- **Training Time:** 10-15 minutes (with pre-generated data)
- **Inference Time:** < 1 second per comparison

## Troubleshooting

### CUDA Out of Memory

If you get GPU memory errors:

```python
# In traffic_forecast_model.py, reduce batch size
class Config:
    BATCH_SIZE = 50  # Default is 100
```

Or force CPU usage:
```python
device = torch.device('cpu')
```

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Model Not Found

If you see "Model not found" errors:
```bash
# Train the model first
python traffic_forecast_model.py
```

### Slow Performance on CPU

Expected times on CPU:
- Embedding generation: ~20-30 minutes
- Training: < 1 minute
- Inference: 2-3 seconds per prediction

To speed up, use a GPU or reduce data size for testing.

## File Structure After Setup

```
anyword/
├── traffic_forecast_model.py       # Main model
├── example_usage.py                # Examples
├── requirements.txt                # Dependencies
├── README.md                       # Full documentation
├── QUICKSTART.md                   # This file
├── traffic_model.pkl              # Trained model (after training)
├── daily_df.csv                   # Data with LLM forecasts (optional)
└── Metro_Interstate_Traffic_Volume.csv.gz  # Original data
```

## What's Next?

1. **Read the full README:** Detailed design choices and evaluation
2. **Try different forecasts:** See how weather affects traffic
3. **Analyze limitations:** Understand when the model works well
4. **Explore improvements:** Ideas in README for better performance

## Quick Reference

```bash
# Train model
python traffic_forecast_model.py

# Run examples
python example_usage.py

# Interactive comparison
python example_usage.py interactive

# Single forecast
python example_usage.py single

# Load and use in Python
python -c "from traffic_forecast_model import TrafficForecastModel; \
           model = TrafficForecastModel(); \
           model.load_model('traffic_model.pkl')"
```

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review the notebook `anyword_task.ipynb` for development process
3. Open an issue in the repository

Happy forecasting! 🚗☀️🌧️
