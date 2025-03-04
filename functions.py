import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant
import yfinance as yf
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns

def get_stock(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']
    data = data.pct_change().dropna()
    return data

def get_benchmark(benchmark, start, end):
    benchmark_data = yf.download(benchmark, start=start, end=end)['Close']
    benchmark_data = benchmark_data.pct_change().dropna()
    return benchmark_data

def volatility(ticker_data):
    returns = ticker_data.dropna()
    vol = returns.std()
    return vol

def beta(ticker_data, benchmark_data):
    returns = pd.concat([ticker_data, benchmark_data], axis=1).dropna()
    returns.columns = ['Stock', 'Market']
    covariance = np.cov(returns['Stock'], returns['Market'])[0, 1]
    market_variance = np.var(returns['Market'])
    beta_val = covariance / market_variance
    return beta_val

def treynor_ratio(ticker_data, benchmark_data, risk_free_rate, beta_val):
    returns = pd.concat([ticker_data, benchmark_data], axis=1).dropna()
    returns.columns = ['Stock', 'Market']
    excess_returns = returns['Stock'] - risk_free_rate
    if beta_val is None:
        beta_val = beta(ticker_data, benchmark_data)
    treynor = (excess_returns.mean() * np.sqrt(252)) / beta_val
    return treynor

def sharpe_ratio(ticker_data, benchmark_data, risk_free_rate):
    returns = pd.concat([ticker_data, benchmark_data], axis=1).dropna()
    returns.columns = ['Stock', 'Market']
    excess_returns = returns['Stock'] - risk_free_rate
    sd = np.std(returns['Stock'])
    sharpe = (excess_returns.mean() / sd) * np.sqrt(252)
    return sharpe

def alpha(ticker_data, benchmark_data):
    # Concatenate the two return series, replace inf with NaN and drop all rows with NaNs.
    returns_data = pd.concat([ticker_data, benchmark_data], axis=1)
    returns_data = returns_data.replace([np.inf, -np.inf], np.nan).dropna()
    returns_data.columns = ['Stock', 'Market']
    X = add_constant(returns_data['Market'])
    y = returns_data['Stock']
    model = sm.OLS(y, X).fit()
    alpha_val = model.params['const']
    return alpha_val

def distribution_plot_base64(ticker_data):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(ticker_data, bins=50, element="step", common_norm=False, multiple="stack", alpha=1, ax=ax)
    ax.set_title("Distribution of Daily Returns")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return plot_data

def get_risk_free_rate(start, end):
    risk_free_data = yf.download("^IRX", start=start, end=end)
    risk_free_series = risk_free_data['Close'] / 100  # Convert percentage to decimal
    daily_risk_free_series = risk_free_series / 252
    return daily_risk_free_series.mean()  # Removed [0] to avoid FutureWarning

# Portfolio analysis functions

def portfolio_beta(daily_returns, benchmark_returns, weights=None):
    if weights is None:
        weights = np.repeat(1/len(daily_returns.columns), len(daily_returns.columns))
    portfolio_daily_returns = daily_returns.dot(weights)
    df = pd.concat([portfolio_daily_returns, benchmark_returns], axis=1).dropna()
    df.columns = ['Portfolio', 'Benchmark']
    covariance = np.cov(df['Portfolio'], df['Benchmark'])[0, 1]
    market_variance = np.var(df['Benchmark'])
    beta_val = covariance / market_variance
    return beta_val

def portfolio_volatility(daily_returns, weights=None):
    if weights is None:
        weights = np.repeat(1/len(daily_returns.columns), len(daily_returns.columns))
    portfolio_daily_returns = daily_returns.dot(weights)
    vol = np.std(portfolio_daily_returns) * np.sqrt(252)
    return vol

def portfolio_sharpe_ratio(daily_returns, risk_free_rate, weights=None):
    if weights is None:
        weights = np.repeat(1/len(daily_returns.columns), len(daily_returns.columns))
    portfolio_daily_returns = daily_returns.dot(weights)
    excess_returns = portfolio_daily_returns - risk_free_rate
    sharpe = (np.mean(excess_returns) / np.std(portfolio_daily_returns)) * np.sqrt(252)
    return sharpe

def portfolio_treynor_ratio(daily_returns, benchmark_returns, risk_free_rate, weights=None):
    if weights is None:
        weights = np.repeat(1/len(daily_returns.columns), len(daily_returns.columns))
    portfolio_daily_returns = daily_returns.dot(weights)
    df = pd.concat([portfolio_daily_returns, benchmark_returns], axis=1).dropna()
    df.columns = ['Portfolio', 'Benchmark']
    covariance = np.cov(df['Portfolio'], df['Benchmark'])[0, 1]
    beta_val = covariance / np.var(df['Benchmark'])
    excess_returns = df['Portfolio'] - risk_free_rate
    treynor = (excess_returns.mean() * np.sqrt(252)) / beta_val
    return treynor

def portfolio_alpha(daily_returns, benchmark_returns, weights=None):
    if weights is None:
        weights = np.repeat(1/len(daily_returns.columns), len(daily_returns.columns))
    portfolio_daily_returns = daily_returns.dot(weights)
    df = pd.concat([portfolio_daily_returns, benchmark_returns], axis=1).dropna()
    df.columns = ['Portfolio', 'Benchmark']
    X = add_constant(df['Benchmark'])
    y = df['Portfolio']
    model = sm.OLS(y, X).fit()
    alpha_val = model.params['const']
    return alpha_val
