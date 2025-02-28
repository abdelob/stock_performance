
import pandas as pd
import yfinance as yf
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels as sm
from statsmodels.tools import add_constant


def get_stock(tickers,start,end):
  data = yf.download(tickers,start= start,end=end)['Close']
  data = data.pct_change().dropna()
  data = data.dropna()
  return data


def get_benchmark(benchmark,start,end):
  benchmark = yf.download(benchmark,start=start,end=end)['Close']
  benchmark = benchmark.pct_change().dropna()
  benchmark = benchmark.dropna()
  return benchmark

def get_risk_free_rate(start,end):
    risk_free_data = yf.download("^IRX",start=start,end=end)

    # Extract the 'Adj Close' column, which represents the annualized risk-free rate in percentage
    risk_free_series = risk_free_data['Close'] / 100  # Convert percentage to decimal

    # Convert to daily risk-free rate assuming 252 trading days per year
    daily_risk_free_series = risk_free_series / 252

    return daily_risk_free_series.mean()[0]



def distribution_plot_base64(ticker_data):
    """
    Creates a Seaborn histogram for daily returns (could be 1+ columns)
    and returns the plot as a base64-encoded PNG string.
    """
    sns.set_style("whitegrid")

    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot: if ticker_data has multiple columns, Seaborn will create stacked histograms
    sns.histplot(ticker_data, bins=50, element="step", 
                 common_norm=False, multiple="stack", alpha=1, ax=ax)

    ax.set_title("Distribution of Daily Returns")
    ax.set_xlabel("Daily Return (as decimal)")
    ax.set_ylabel("Frequency")

    # If multiple columns, add a legend. If only one, it won't break anything.
    if isinstance(ticker_data, pd.DataFrame):
        ax.legend(ticker_data.columns, title="Stocks")

    plt.tight_layout()

    # Convert figure to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return plot_data




# Beta


import numpy as np

def beta(ticker_data,benchmark_data):
  returns = pd.concat([ticker_data, benchmark_data], axis=1)
  returns.columns = ['Stock', 'Market']
  returns = returns.dropna()

  covariance = np.cov(returns['Stock'], returns['Market'])[0][1]
  market_variance = np.var(returns['Market'])
  beta = covariance / market_variance

  return round(beta,2)


# Treynor Ratio

def treynor_ratio(ticker_data, benchmark_data, risk_free_rate, beta):
    returns = pd.concat([ticker_data, benchmark_data], axis=1)
    returns.columns = ['Stock', 'Market']
    returns = returns.dropna()
    excess_returns = returns['Stock'] - risk_free_rate
    treynor = (excess_returns.mean() / beta) * np.sqrt(252)
    return round(treynor, 2)



def sharpe_ratio(ticker_data,benchmark_data,risk_free_rate):
  returns = pd.concat([ticker_data, benchmark_data], axis=1)
  returns.columns = ['Stock', 'Market']
  returns = returns.dropna()
  excess_returns = returns['Stock'] - risk_free_rate
  sd = np.std(returns['Stock'])
  sharpe_ratio = ((excess_returns.mean() - risk_free_rate) / sd) * np.sqrt(252)
  return round(sharpe_ratio,2)

# Volatility

def volatility(ticker_data):
  returns = ticker_data.dropna()
  volatility = returns.std()
  return round(volatility,2)[0]


import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant

def jensen_alpha(ticker_data, benchmark_data):
    returns_data = pd.concat([ticker_data, benchmark_data], axis=1)
    returns_data.columns = ['Stock', 'Market']
    x = returns_data['Market']
    y = returns_data['Stock']
    X = add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha = model.params['const']
    return alpha



# CSV

def data_csv(ticker, start, end, csv_filename):
    try:
        # Fetch historical data
        data = yf.download(ticker, start=start, end=end)['Close']

        # Save data to CSV
        data.to_csv(csv_filename)

        print(f"Stock data for {ticker} has been saved to {csv_filename}")
        return csv_filename
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



