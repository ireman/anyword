# Traffic Forecast Comparison Model - Technical Report

## Overview

This project implements a machine learning system that compares two weather forecast texts and predicts which will result in higher traffic volume, along with the percentage difference. The model was developed as a solution to predict traffic patterns based on natural language weather descriptions.

## Implementation

### Dataset and Preprocessing

The model uses the **Metro Interstate Traffic Volume dataset** from the UCI Machine Learning Repository, containing ~48,000 hourly traffic observations from Minneapolis-St Paul, MN (2012-2018). The data was aggregated from hourly to daily level, calculating:
- Total and normalized traffic volume (average hourly traffic per day)
- Weather aggregations (min/max temperature, average cloud cover, precipitation totals, mode weather description)

### Text Generation Strategy

Since no naturally written weather forecast dataset was readily available, two types of textual forecasts were generated:

1. **Simple Forecasts**: Structured templates generated directly from aggregated numerical weather data
2. **LLM-Enhanced Forecasts**: Detailed narrative descriptions generated using OpenAI GPT-3.5-turbo, which synthesizes hourly weather progressions into natural language forecasts

### Model Architecture

Given time and computational constraints, a pragmatic two-stage approach was adopted:

1. **Text Embedding**: DistilBERT transformer model converts forecast texts into 768-dimensional semantic embeddings
2. **Regression**: Ridge regression (L2 regularization, α=1.0) trained on concatenated embeddings from both forecast types (1,536 dimensions total) to predict normalized traffic volume

### Results

- **Test RMSE**: ~613 people/hour (~17% relative error)
- **Training Time**: ~15 minutes on GPU with pre-generated LLM forecasts
- **Output Format**: Winner (A or B) + percentage difference

**Example:**
```
Forecast A: "Clear skies with temperatures in the lower 60s"
Forecast B: "Cloudy with a small chance of rain and temperatures in the lower 70s"
Result: A, 7.32%
```

## Limitations

The current model has several notable limitations that constrain its predictive accuracy:

### 1. Lack of Temporal Context
The model relies solely on weather descriptions without crucial contextual information such as:
- Day of week (weekday vs. weekend patterns)
- Time of year (seasonal variations)
- Holidays and special events
- Rush hour vs. off-peak distinctions

These temporal factors significantly influence traffic patterns but are not captured by weather forecasts alone.

### 2. Dataset Constraints
No rich, naturally-written weather forecast dataset was available for this task. The approach of converting tabular weather data into text using an LLM is a workaround, but ideally one would use:
- Real weather forecast transcripts from meteorologists
- Historical forecast archives with natural language descriptions
- Multi-modal data combining forecast text with structured environmental features

### 3. Single-City Limitation
The model is trained exclusively on Minneapolis-St Paul data and may not generalize to cities with different:
- Climate patterns
- Urban infrastructure
- Traffic behaviors

## Future Research Directions

With additional time and resources, several advanced approaches could substantially improve model performance:

### 1. Temporal Feature Engineering
Incorporate time-based features to capture traffic patterns:
- Day of week and time of day encoding
- Holiday indicators and special event calendars
- Seasonal patterns (cyclical encoding)
- Traffic history (lagged features from previous days)

**Expected Impact**: 20-30% reduction in RMSE by accounting for non-weather factors that drive traffic.

### 2. Fine-Tuned Language Models
Move beyond using transformers solely as embedding generators:
- Fine-tune a language model (e.g., claude, GPT) specifically for the weather-to-traffic prediction task
- Train end-to-end: text input → direct traffic volume output
- Alternatively, use knowledge distillation to create a specialized smaller model

**Advantage**: The model learns task-specific representations rather than generic semantic embeddings.

### 3. Hybrid Architecture
Combine strengths of language models and structured prediction:
- LM extracts structured features from weather text (temperature ranges, precipitation likelihood, condition severity)
- Features feed into specialized regression model (e.g., Gradient Boosting, Neural Network)
- Allows interpretability of which weather aspects most influence predictions

### 4. Multi-City Transfer Learning
Build a universal weather-traffic model:
- Train on multiple cities simultaneously to learn general patterns
- Fine-tune for specific cities with limited local data
- Enables deployment to new cities with minimal training data

### 5. Richer Data Sources
Seek out or construct better datasets:
- Historical weather forecast archives (as broadcast/written by meteorologists)
- Paired datasets: forecast text + actual observed weather + traffic outcomes
- Multi-modal inputs: text forecasts + weather radar images + historical traffic patterns

### 6. Uncertainty Quantification
Implement probabilistic predictions:
- Bayesian regression or ensemble methods
- Output prediction intervals: "3,000 ± 400 people/hour (90% confidence)"
- Helps users understand reliability of predictions

### 7. Real-Time Integration
Move from historical analysis to operational system:
- Ingest live weather forecasts from APIs
- Integrate with real-time traffic monitoring
- Continuous learning from new data
- Deploy as microservice for traffic management systems

## Computational and Data Requirements

The current approach was chosen specifically due to constraints in:
- **Computational resources**: No access to large-scale GPU clusters for fine-tuning large language models
- **Data availability**: No pre-existing dataset of natural language weather forecasts paired with traffic data
- **Time constraints**: Needed to deliver a working solution quickly

A production-grade system would benefit from:
- Multi-GPU infrastructure for model training
- Larger datasets spanning multiple cities and years
- Access to proprietary weather forecast data
- Integration with city traffic management systems


---

**Technical Stack**:
- Language: Python 3.8+
- ML Frameworks: PyTorch, Transformers (Hugging Face), Scikit-learn
- NLP Model: DistilBERT (distilbert-base-uncased)
- LLM API: OpenAI GPT-3.5-turbo
- Dataset: UCI Machine Learning Repository - Metro Interstate Traffic Volume

**Code**: Available at `traffic_forecast_model.py` (~500 lines)
