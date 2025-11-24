# Traffic Volume Prediction from Weather Forecasts

A machine learning model that predicts traffic volume differences based on weather forecast descriptions. Given two weather forecast texts, the model predicts which forecast will result in more traffic and by what percentage.

## Problem Statement

**Input:** Two weather forecast texts (A and B)
**Output:** Which forecast has more traffic and the percentage difference

**Example:**
- Forecast A: "Clear skies with temperatures in the lower 60s"
- Forecast B: "Cloudy with a small chance of rain and temperatures in the lower 70s"
- Output: `A, 7.21%` (Forecast A will have 7.21% more traffic)

## Dataset

**Source:** [Metro Interstate Traffic Volume Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz)

**Location:** Minneapolis-St Paul, MN, USA
**Time Period:** 2012-2018
**Records:** ~48,000 hourly observations
**Aggregation:** Daily level (1,860 days)

**Features:**
- Weather conditions (temperature, rain, snow, clouds)
- Weather descriptions (text)
- Hourly traffic volume

## Design Choices

### 1. Data Aggregation Strategy

**Choice:** Aggregate hourly data to daily level

**Rationale:**
- Weather forecasts are typically given for full days, not specific hours
- Reduces noise from hourly variations
- Creates more stable target variable (normalized traffic = sum/count)
- Matches real-world use case where specific time-of-day is not provided

**Aggregation:**
- Traffic: Sum of hourly volumes
- Temperature: Min and Max for the day
- Weather description: Mode (most frequent)
- Rain/Snow: Sum of accumulation
- Clouds: Average percentage

### 2. Text Representation Approach

**Choice:** Dual-text encoding using BERT embeddings

**Two types of forecast texts:**

1. **Simple Forecasts** - Structured summaries from numerical data:
   ```
   "Daily weather is sky is clear. Clouds average at 29%.
   Conditions: dry, no snow. Min temp approx 54 degrees,
   Max temp approx 69 degrees."
   ```

2. **LLM-Enhanced Forecasts** - Natural language narratives:
   ```
   "The day will start with scattered clouds and temperatures
   in the upper 50s. Cloud cover will increase through the
   morning, reaching overcast conditions by midday..."
   ```

**Rationale:**
- **Simple forecasts** provide direct numerical grounding
- **LLM forecasts** capture temporal patterns and contextual descriptions
- **Combined embeddings** (concatenated) leverage both structured and natural language information
- More robust to different input styles users might provide

### 3. Embedding Model

**Choice:** DistilBERT (distilbert-base-uncased)

**Rationale:**
- Pretrained on large text corpus - understands weather terminology
- Contextual embeddings - captures semantic meaning
- Fast inference compared to full BERT
- 768-dimensional embeddings capture rich textual features
- Proven effectiveness for text similarity and semantic understanding

**Alternative considered:** Sentence-BERT (faster, similar performance for shorter texts)

### 4. Regression Model

**Choice:** Ridge Regression (L2 regularization)

**Rationale:**
- Effective for high-dimensional embeddings (768 or 1536 dimensions)
- Prevents overfitting through regularization
- Fast training and inference
- Interpretable linear relationship
- Surprisingly competitive with more complex models for embedding-based regression

**Alpha = 1.0** - Balanced regularization strength

**Alternatives considered:**
- Random Forest: Good but slower, less effective with high-dim embeddings
- Gradient Boosting: Similar performance, much slower training
- Neural Network: Potential overfitting risk with limited data

### 5. Target Variable

**Choice:** Normalized traffic volume (average hourly traffic per day)

**Formula:** `traffic_volume_sum / original_entries_count`

**Rationale:**
- Accounts for missing hours in some days
- More stable and interpretable than raw sums
- Represents "typical hourly traffic" for that day
- Better generalization for percentage comparisons

## Model Architecture

```
Input: Weather Forecast Text
    ↓
DistilBERT Tokenizer
    ↓
DistilBERT Model (frozen)
    ↓
CLS Token Embedding (768-dim)
    ↓
[Optional: Concatenate with LLM forecast embedding]
    ↓
Ridge Regression (α=1.0)
    ↓
Output: Predicted Traffic Volume (people/hour)
```

## Performance Evaluation

### Metrics

**Test Set RMSE:** ~610-615 people/hour
**Test Set MAE:** ~450-480 people/hour

### Interpretation

- **Context:** Average hourly traffic is ~3,500 people/hour
- **Relative Error:** ~17-18% RMSE
- **MAE:** About 13-14% mean absolute error

### Performance Analysis

**Strengths:**
- ✅ Captures general weather-traffic relationships
- ✅ Correctly predicts severe weather reduces traffic
- ✅ Handles diverse natural language inputs
- ✅ Stable across different forecast writing styles

**Limitations:**
- ❌ 17% error means predictions can be off by 500-700 people/hour
- ❌ No temporal information (day of week, season, holidays)
- ❌ Cannot distinguish rush hour vs. off-peak patterns
- ❌ Limited training data (only 1,860 days)

### Example Predictions

| Forecast | Predicted Traffic | Actual Pattern |
|----------|------------------|----------------|
| Clear, 60°F | ~3,000-3,200 people/hour | Moderate-High |
| Heavy snow, 30°F | ~2,400-2,600 people/hour | Lower |
| Light rain, 70°F | ~2,900-3,100 people/hour | Moderate |

## Installation

```bash
# Clone repository
git clone <repository-url>
cd anyword

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```python
from traffic_forecast_model import main

# Train model with default settings
main()
```

This will:
1. Download and load traffic data
2. Aggregate to daily level
3. Generate embeddings
4. Train Ridge regression model
5. Save model to `traffic_model.pkl`
6. Display performance metrics and examples

### Making Predictions

#### Option 1: Python API

```python
from traffic_forecast_model import TrafficForecastModel

# Load pre-trained model
model = TrafficForecastModel()
model.load_model('traffic_model.pkl')

# Compare two forecasts
forecast_a = "Clear skies with temperatures in the lower 60s"
forecast_b = "Cloudy with a small chance of rain and temperatures in the lower 70s"

winner, pct_diff, pred_a, pred_b = model.compare_forecasts(forecast_a, forecast_b)

print(f"Result: {winner}, {pct_diff:.2f}%")
# Output: Result: A, 7.21%
```

#### Option 2: Command Line

```bash
# Train model
python traffic_forecast_model.py

# Interactive prediction
python traffic_forecast_model.py predict
```

### Using Pre-generated LLM Forecasts

If you have the `daily_df.csv` file with pre-generated LLM forecasts:

```python
# Place daily_df.csv in the working directory
# The script will automatically detect and use it
main()
```

### Generating New LLM Forecasts (Optional)

Uncomment the OpenAI section in `main()` and set your API key:

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'

# Or export in shell:
# export OPENAI_API_KEY='your-key-here'
```

**Note:** Generating LLM forecasts for 1,860 days costs ~$5-10 in OpenAI API credits.

## File Structure

```
anyword/
├── traffic_forecast_model.py    # Main model code
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── anyword_task.ipynb           # Original exploration notebook
├── traffic_model.pkl            # Saved model (after training)
├── daily_df.csv                 # Pre-generated data with LLM forecasts
└── Metro_Interstate_Traffic_Volume.csv.gz  # Original dataset
```

## Limitations and Future Improvements

### Current Limitations

1. **No Temporal Context**
   - Model doesn't know if it's a weekday/weekend
   - Doesn't account for holidays or special events
   - Ignores seasonal patterns beyond weather

2. **Single City**
   - Trained only on Minneapolis-St Paul data
   - May not generalize to other cities with different:
     - Weather patterns
     - Traffic behaviors
     - Infrastructure

3. **No Time-of-Day Information**
   - Predicts daily average, not peak hours
   - Cannot distinguish morning vs evening rush

4. **Limited Training Data**
   - Only 5+ years of data
   - Weather patterns may not cover all extremes
   - Traffic patterns may have changed over time

5. **No Confidence Intervals**
   - Point predictions only
   - No uncertainty quantification

### Potential Improvements (With More Time/Resources)

#### 1. Temporal Features Engineering
```python
# Add these features:
- Day of week (one-hot encoded)
- Month (cyclical encoding: sin/cos)
- Is_holiday (binary)
- Is_weekend (binary)
- Days_since_snow (capture road conditions)
```

**Expected Impact:** Could reduce RMSE by 20-30%

#### 2. Multi-Task Learning
Train model to predict:
- Total daily traffic (current)
- Morning rush hour traffic (6-9 AM)
- Evening rush hour traffic (4-7 PM)
- Off-peak traffic

**Benefit:** More actionable predictions

#### 3. Ensemble Methods
Combine multiple models:
- BERT embeddings + Ridge
- Weather features + Gradient Boosting
- Time features + Neural Network

**Expected Impact:** 10-15% RMSE improvement

#### 4. Better Text Encoding
- **Sentence-BERT:** Optimized for semantic similarity
- **Weather-specific BERT:** Fine-tuned on weather corpus
- **Multi-modal:** Combine text + numerical weather features directly

#### 5. Uncertainty Quantification
- Implement Bayesian Ridge Regression
- Or use quantile regression for prediction intervals
- Would output: "3000 ± 400 people/hour (90% confidence)"

#### 6. Real-Time Weather API Integration
- Fetch actual forecasts from weather services
- Compare "as-written" forecasts vs model predictions
- Continuous learning from new data

#### 7. Cross-City Transfer Learning
- Train on multiple cities
- Learn general weather-traffic patterns
- Fine-tune for specific cities
- Build a universal weather-traffic model

#### 8. Explainability Features
Add SHAP or LIME to explain:
- Which words in forecast most influence predictions
- How much each weather aspect contributes
- Provide reasoning: "High traffic predicted because..."

## Mathematical Formulation

### Objective Function

Minimize:
```
L(θ) = (1/n) Σ(yᵢ - f(xᵢ; θ))² + λ||θ||²
```

Where:
- `yᵢ` = normalized traffic volume for day i
- `xᵢ` = BERT embedding of forecast text
- `f(x; θ)` = linear predictor (Ridge regression)
- `λ` = regularization strength (α = 1.0)

### Comparison Function

For two forecasts A and B:

```
pred_A = θᵀ · BERT(text_A)
pred_B = θᵀ · BERT(text_B)

if pred_A > pred_B:
    winner = A
    pct_diff = ((pred_A - pred_B) / pred_B) × 100
else:
    winner = B
    pct_diff = ((pred_B - pred_A) / pred_A) × 100
```

## References

- **Dataset:** Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

- **DistilBERT:** Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

- **Ridge Regression:** Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55-67.

## License

MIT License - See LICENSE file for details

## Contact

For questions or feedback, please open an issue in the repository.
