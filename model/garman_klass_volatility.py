import pandas as pd
import numpy as np


def garman_klass_volatility(high, low, open_price, close):
    """
    Calculate Garman-Klass volatility proxy

    The Garman-Klass estimator is more efficient than simple return-based volatility
    as it uses high, low, open, and close prices.

    Formula: σ² = 0.5 * (ln(H/L))² - (2*ln(2) - 1) * (ln(C/O))²

    Parameters:
    high: high price
    low: low price
    open_price: opening price
    close: closing price

    Returns:
    volatility estimate
    """
    # Avoid division by zero and log of negative numbers
    high = np.maximum(high, low + 1e-8)  # Ensure high > low
    open_price = np.maximum(open_price, 1e-8)  # Ensure open > 0
    close = np.maximum(close, 1e-8)  # Ensure close > 0

    # Calculate log returns
    log_hl = np.log(high / low)
    log_co = np.log(close / open_price)

    # Garman-Klass formula
    volatility_squared = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)

    # Return volatility (square root)
    return np.sqrt(np.maximum(volatility_squared, 0))


def add_garman_klass_volatility_to_data(
    file_path="data/merged_data_with_realised_volatility.csv",
):
    """
    Add Garman-Klass volatility to existing data

    Parameters:
    file_path: path to the CSV file with merged data
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Calculate Garman-Klass volatility
    df["garman_klass_volatility"] = garman_klass_volatility(
        df["High"], df["Low"], df["Open"], df["Close"]
    )

    # Calculate rolling Garman-Klass volatility (21-day window)
    df["garman_klass_realised_volatility"] = (
        df["garman_klass_volatility"].rolling(window=21).mean()
    )

    # Save the updated data
    df.to_csv(file_path, index=False)

    print("Garman-Klass volatility added successfully!")
    print(
        f"Columns added: 'garman_klass_volatility', 'garman_klass_realised_volatility'"
    )

    # Show comparison
    print("\nComparison of volatility measures (last 5 rows):")
    print("Original realised volatility (using returns):")
    print(df["realised_volatility"].tail())
    print("\nGarman-Klass realised volatility (using OHLC):")
    print(df["garman_klass_realised_volatility"].tail())

    return df


if __name__ == "__main__":
    # Add Garman-Klass volatility to the data
    df_updated = add_garman_klass_volatility_to_data()
