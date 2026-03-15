import pandas as pd
import os

#  DATA FOLDER


data_folder = "data"

crypto_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

print(f"Loaded {len(crypto_files)} cryptocurrencies successfully!")

crypto_data = {}

# DATA PREPROCESSING


for file in crypto_files:

    file_path = os.path.join(data_folder, file)
    coin_name = file.replace(".csv", "")

    df = pd.read_csv(file_path)

    # Rename "Price" column to "Date"
    df.rename(columns={"Price": "Date"}, inplace=True)

    # Keep only Date and Close
    df = df[["Date", "Close"]]

    # Convert data types
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Drop missing values
    df = df.dropna(subset=["Date", "Close"])

    # Sort and set index
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    # Calculate Daily Return
    df["Daily Return"] = df["Close"].pct_change()

    # Remove extreme outliers
    df = df[(df["Daily Return"] < 5) & (df["Daily Return"] > -5)]

    # Drop remaining NaN
    df = df.dropna()

    crypto_data[coin_name] = df

print("Data preprocessing completed successfully!")

# VOLATILITY


print("\nVolatility (Risk) of each coin:")

for coin in crypto_data:
    volatility = crypto_data[coin]["Daily Return"].std()
    print(f"{coin}: {volatility:.5f}")


# TOTAL RETURN


print("\nTotal Return of each coin:")

total_returns = {}   # ✅ ADD THIS

for coin in crypto_data:

    df = crypto_data[coin]

    if df.empty:
        continue

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]

    total_return = (end_price - start_price) / start_price

    total_returns[coin] = total_return * 100   # ✅ STORE %

    print(f"{coin}: {total_return:.2%}")



# INVESTMENT SIMULATION


initial_investment = 10000
print("\nFinal Value of 10,000 Investment:")


for coin in crypto_data:

    df = crypto_data[coin]

    if df.empty:
        continue

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]

    final_value = initial_investment * (end_price / start_price)

    print(f"{coin}: Rs. {final_value:,.2f}")

import numpy as np

print("\nCAGR (Annualized Return):")

for coin, df in crypto_data.items():
    
    df = df.dropna()
    
    if len(df) > 1:
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        
        num_years = (df.index[-1] - df.index[0]).days / 365
        
        if start_price > 0 and num_years > 0:
            cagr = ((end_price / start_price) ** (1 / num_years)) - 1
            print(f"{coin}: {cagr*100:.2f}%")

print("\nRisk-Adjusted Return (CAGR / Volatility):")

risk_adjusted = {}

for coin, df in crypto_data.items():
    
    df = df.dropna()
    
    if len(df) > 1:
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        
        num_years = (df.index[-1] - df.index[0]).days / 365
        
        if start_price > 0 and num_years > 0:
            cagr = ((end_price / start_price) ** (1 / num_years)) - 1
            
            volatility = df['Close'].pct_change().std()
            
            if volatility and not np.isnan(volatility):
                score = cagr / volatility
                risk_adjusted[coin] = score

# Sort by best score
sorted_scores = sorted(risk_adjusted.items(), key=lambda x: x[1], reverse=True)

print("\nTop 5 Risk-Adjusted Coins:")
for coin, score in sorted_scores[:5]:
    print(f"{coin}: {score:.2f}")

print("\nWorst 5 Risk-Adjusted Coins:")
for coin, score in sorted_scores[-5:]:
    print(f"{coin}: {score:.2f}")

print("\nTop 5 Coins by Total Return:")

sorted_returns = sorted(total_returns.items(), key=lambda x: x[1], reverse=True)

for coin, ret in sorted_returns[:5]:
    print(f"{coin}: {ret:.2f}%")


# MAXIMUM DRAWDOWN


print("\nMaximum Drawdown (Worst Crash):")

for coin, df in crypto_data.items():

    if df.empty:
        continue

    cumulative = (1 + df["Daily Return"]).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    max_drawdown = drawdown.min()

    print(f"{coin}: {max_drawdown:.2%}")


# EQUAL-WEIGHT PORTFOLIO


print("\nEqual-Weight Portfolio Analysis:")

initial_portfolio = 10000
num_coins = len(crypto_data)
investment_per_coin = initial_portfolio / num_coins

portfolio_final_value = 0

for coin, df in crypto_data.items():
    
    if df.empty:
        continue
    
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    
    final_value = investment_per_coin * (end_price / start_price)
    portfolio_final_value += final_value

portfolio_profit = portfolio_final_value - initial_portfolio

print(f"Initial Investment: Rs. {initial_portfolio:,.2f}")
print(f"Final Portfolio Value: Rs. {portfolio_final_value:,.2f}")
print(f"Total Profit: Rs. {portfolio_profit:,.2f}")



# PORTFOLIO CAGR


print("\nPortfolio CAGR & Risk Metrics:")

years = (crypto_data["BTC-USD"].index[-1] - crypto_data["BTC-USD"].index[0]).days / 365

portfolio_cagr = ((portfolio_final_value / initial_portfolio) ** (1 / years) - 1) * 100

print(f"Portfolio CAGR: {portfolio_cagr:.2f}%")

# PORTFOLIO VOLATILITY


import pandas as pd

returns_df = pd.DataFrame()

for coin, df in crypto_data.items():
    if not df.empty:
        returns_df[coin] = df["Daily Return"]

portfolio_daily_return = returns_df.mean(axis=1)

portfolio_volatility = portfolio_daily_return.std()

print(f"Portfolio Volatility: {portfolio_volatility:.5f}")

# PORTFOLIO RISK-ADJUSTED RETURN


portfolio_risk_adjusted = (portfolio_cagr / 100) / portfolio_volatility

print(f"Portfolio Risk-Adjusted Return: {portfolio_risk_adjusted:.2f}")



# FORECASTING SECTION (ARIMA MODEL)

# ARIMA FORECASTING - BTC


btc_price = crypto_data["BTC-USD"]["Close"].dropna()

print(btc_price.head())
print(type(btc_price))

from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(btc_price)

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])


btc_diff = btc_price.diff().dropna()

# Check stationarity again
adf_result_diff = adfuller(btc_diff)

print("ADF Statistic (After Differencing):", adf_result_diff[0])
print("p-value (After Differencing):", adf_result_diff[1])


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(btc_price, order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())


# 30-Day Forecast


forecast = model_fit.forecast(steps=30)

print(forecast)


# BTC FORECAST EXPORT 


btc_df = crypto_data["BTC-USD"]

btc_actual = btc_df["Close"]

# Generate 30-day forecast (if not already done)
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Create future date index
last_date = btc_actual.index[-1]
forecast_index = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_steps,
    freq='D'
)

forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Forecast Price": forecast.values
})

actual_df = pd.DataFrame({
    "Date": btc_actual.index,
    "Actual Price": btc_actual.values
})

btc_combined = pd.merge(actual_df, forecast_df, on="Date", how="outer")

btc_combined.to_csv("btc_forecast.csv", index=False)

print("btc_forecast.csv exported successfully")

# PORTFOLIO OPTIMIZATION - SETUP


import pandas as pd
import numpy as np

# Collect daily returns of all coins
returns_dict = {}

for coin, df in crypto_data.items():
    if 'Close' in df.columns:
        returns = df['Close'].pct_change().dropna()
        returns_dict[coin] = returns

# Combine into one DataFrame
returns_df = pd.DataFrame(returns_dict)
returns_df = returns_df.dropna(how='all')   # Drop only rows where all are NaN

print("Returns DataFrame Shape:", returns_df.shape)


# MEAN RETURNS & COVARIANCE MATRIX


mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

print("Mean Returns:")
print(mean_returns.head())

print("\nCovariance Matrix Shape:", cov_matrix.shape)


# PORTFOLIO OPTIMIZATION - 


from scipy.optimize import minimize

# Drop coins with NaN
returns_df = returns_df.dropna(axis=1)
print("Cleaned Returns Shape:", returns_df.shape)

# Recalculate mean & covariance AFTER cleaning
mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

# Annualize
trading_days = 365
mean_returns_annual = mean_returns * trading_days
cov_matrix_annual = cov_matrix * trading_days

# Number of assets AFTER cleaning
num_assets = len(mean_returns_annual)

# Portfolio performance function
def portfolio_performance(weights):
    returns = np.dot(weights, mean_returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
    return returns, volatility

# Negative Sharpe Ratio
def negative_sharpe(weights):
    returns, volatility = portfolio_performance(weights)
    return -returns / volatility

# Constraints & bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))

# Equal weight start
initial_weights = np.ones(num_assets) / num_assets

# Optimize
result = minimize(negative_sharpe,
                  initial_weights,
                  method='SLSQP',
                  bounds=bounds,
                  constraints=constraints)

optimal_weights = result.x

print("Optimal Portfolio Weights:")
for coin, weight in zip(returns_df.columns, optimal_weights):
    if weight > 0.01:
        print(f"{coin}: {weight:.4f}")
# Annualize returns and covariance
trading_days = 365  # crypto trades every day

mean_returns_annual = mean_returns * trading_days
cov_matrix_annual = cov_matrix * trading_days

# COIN METRICS EXPORT


metrics_list = []

for coin, df in crypto_data.items():
    
    if len(df) > 2:
        start = df["Close"].iloc[0]
        end = df["Close"].iloc[-1]
        years = len(df) / 365
        
        total_return = (end / start - 1) * 100
        cagr = ((end / start) ** (1 / years) - 1) * 100
        volatility = df["Daily Return"].std() * np.sqrt(365)
        sharpe = cagr / (volatility * 100) if volatility != 0 else 0
        
        cumulative = (1 + df["Daily Return"]).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        metrics_list.append({
            "Coin": coin,
            "Total Return (%)": total_return,
            "CAGR (%)": cagr,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_drawdown
        })

metrics_df = pd.DataFrame(metrics_list)

metrics_df.to_csv("coin_metrics.csv", index=False)

print("coin_metrics.csv exported successfully")


# RISK VS RETURN EXPORT


risk_return_df = metrics_df[["Coin", "CAGR (%)", "Volatility"]]

risk_return_df.to_csv("risk_return.csv", index=False)

print("risk_return.csv exported successfully")


# OPTIMIZED WEIGHTS EXPORT


weights_df = pd.DataFrame({
    "Coin": returns_df.columns,
    "Weight": optimal_weights
})

weights_df = weights_df[weights_df["Weight"] > 0]

weights_df.to_csv("optimized_weights.csv", index=False)

print("optimized_weights.csv exported successfully")

# EQUAL WEIGHT PORTFOLIO GROWTH EXPORT


equal_weight_returns = returns_df.mean(axis=1)
equal_cumulative = (1 + equal_weight_returns).cumprod()

equal_growth_df = pd.DataFrame({
    "Date": equal_cumulative.index,
    "Equal Portfolio Growth": equal_cumulative.values
})

equal_growth_df.to_csv("equal_portfolio_growth.csv", index=False)

print("equal_portfolio_growth.csv exported successfully")

# OPTIMIZED PORTFOLIO GROWTH EXPORT


optimized_returns = returns_df.dot(optimal_weights)
optimized_cumulative = (1 + optimized_returns).cumprod()

optimized_growth_df = pd.DataFrame({
    "Date": optimized_cumulative.index,
    "Optimized Portfolio Growth": optimized_cumulative.values
})

optimized_growth_df.to_csv("optimized_portfolio_growth.csv", index=False)

print("optimized_portfolio_growth.csv exported successfully")
