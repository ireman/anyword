"""
Example Usage of Traffic Forecast Model

This script demonstrates how to use the trained model to compare
weather forecasts and predict traffic differences.
"""

from traffic_forecast_model import TrafficForecastModel
import os


def example_comparisons():
    """Run several example forecast comparisons"""

    print("="*70)
    print("TRAFFIC FORECAST MODEL - EXAMPLE USAGE")
    print("="*70)

    # Load pre-trained model
    print("\nLoading model...")
    model = TrafficForecastModel()

    if not os.path.exists('traffic_model.pkl'):
        print("ERROR: Model not found. Please train the model first by running:")
        print("  python traffic_forecast_model.py")
        return

    model.load_model('traffic_model.pkl')
    print("Model loaded successfully!")

    # Define test cases
    test_cases = [
        {
            "name": "Original Task Example",
            "forecast_a": "Clear skies with temperatures in the lower 60s",
            "forecast_b": "Cloudy with a small chance of rain and temperatures in the lower 70s"
        },
        {
            "name": "Extreme Weather Comparison",
            "forecast_a": "Sunny and warm, temperatures in the mid 70s with light winds",
            "forecast_b": "Heavy snowfall expected with accumulation of 6 inches, temperatures below freezing"
        },
        {
            "name": "Rain vs Clear",
            "forecast_a": "Overcast with periods of heavy rain throughout the day",
            "forecast_b": "Morning fog clearing to partly cloudy skies, highs near 65"
        },
        {
            "name": "Temperature Variation",
            "forecast_a": "Cold and clear, temperatures in the 40s",
            "forecast_b": "Warm and humid, temperatures in the 80s"
        },
        {
            "name": "Similar Conditions",
            "forecast_a": "Partly cloudy with highs in the upper 60s",
            "forecast_b": "Mostly sunny with temperatures around 65"
        }
    ]

    # Run comparisons
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*70}")

        forecast_a = test_case['forecast_a']
        forecast_b = test_case['forecast_b']

        print(f"\nForecast A: {forecast_a}")
        print(f"Forecast B: {forecast_b}")

        # Get predictions
        winner, pct_diff, pred_a, pred_b = model.compare_forecasts(forecast_a, forecast_b)

        print(f"\n{'-'*70}")
        print(f"Predicted Traffic:")
        print(f"  Forecast A: {pred_a:.2f} people/hour")
        print(f"  Forecast B: {pred_b:.2f} people/hour")
        print(f"\n{'*'*70}")
        print(f"RESULT: {winner}, {pct_diff:.2f}%")
        print(f"{'*'*70}")
        print(f"Forecast {winner} is expected to have {pct_diff:.2f}% more traffic")


def interactive_mode():
    """Interactive mode for custom forecast comparisons"""

    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model = TrafficForecastModel()

    if not os.path.exists('traffic_model.pkl'):
        print("ERROR: Model not found. Please train the model first.")
        return

    model.load_model('traffic_model.pkl')

    while True:
        print("\n" + "-"*70)
        print("Enter two weather forecasts to compare (or 'quit' to exit):")
        print("-"*70)

        forecast_a = input("\nForecast A: ").strip()
        if forecast_a.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode. Goodbye!")
            break

        forecast_b = input("Forecast B: ").strip()
        if forecast_b.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode. Goodbye!")
            break

        if not forecast_a or not forecast_b:
            print("\nError: Both forecasts must be non-empty. Try again.")
            continue

        # Get predictions
        try:
            winner, pct_diff, pred_a, pred_b = model.compare_forecasts(forecast_a, forecast_b)

            print(f"\n{'='*70}")
            print("PREDICTION RESULTS")
            print("="*70)
            print(f"Forecast A traffic: {pred_a:.2f} people/hour")
            print(f"Forecast B traffic: {pred_b:.2f} people/hour")
            print(f"\n{'*'*70}")
            print(f"RESULT: {winner}, {pct_diff:.2f}%")
            print(f"{'*'*70}")
            print(f"Forecast {winner} is expected to have {pct_diff:.2f}% more traffic")

        except Exception as e:
            print(f"\nError during prediction: {e}")
            print("Please try again with different inputs.")


def single_forecast_prediction():
    """Predict traffic for a single forecast"""

    print("\n" + "="*70)
    print("SINGLE FORECAST PREDICTION")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model = TrafficForecastModel()

    if not os.path.exists('traffic_model.pkl'):
        print("ERROR: Model not found. Please train the model first.")
        return

    model.load_model('traffic_model.pkl')

    print("\nEnter a weather forecast to predict traffic volume:")
    forecast = input("\nForecast: ").strip()

    if not forecast:
        print("Error: Forecast cannot be empty.")
        return

    # Get prediction
    try:
        prediction = model.predict_traffic(forecast)

        print(f"\n{'='*70}")
        print("PREDICTION")
        print("="*70)
        print(f"Forecast: {forecast}")
        print(f"\nPredicted Traffic: {prediction:.2f} people/hour")
        print("="*70)

        # Provide context
        print("\nContext:")
        print(f"  - Average traffic: ~3,500 people/hour")
        if prediction > 3500:
            print(f"  - This forecast predicts ABOVE average traffic")
        elif prediction < 3500:
            print(f"  - This forecast predicts BELOW average traffic")
        else:
            print(f"  - This forecast predicts AVERAGE traffic")

    except Exception as e:
        print(f"\nError during prediction: {e}")


def main():
    """Main function to run examples"""

    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'interactive' or mode == 'i':
            interactive_mode()
        elif mode == 'single' or mode == 's':
            single_forecast_prediction()
        elif mode == 'examples' or mode == 'e':
            example_comparisons()
        else:
            print(f"Unknown mode: {mode}")
            print("\nUsage:")
            print("  python example_usage.py                # Run all examples")
            print("  python example_usage.py examples       # Run all examples")
            print("  python example_usage.py interactive    # Interactive comparison mode")
            print("  python example_usage.py single         # Single forecast prediction")
    else:
        # Default: run examples
        example_comparisons()

        # Ask if user wants to try interactive mode
        print("\n" + "="*70)
        choice = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_mode()


if __name__ == "__main__":
    main()
