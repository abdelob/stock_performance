import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from flask import Flask, render_template, request
import functions

app = Flask(__name__)

def plot_correlation_matrix(ticker_data):
    """
    Generate a correlation matrix plot from ticker_data and return it as a base64-encoded image.
    """
    corr_matrix = ticker_data.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return plot_data

@app.route('/', methods=['GET', 'POST'])
def index():
    # Stock Analysis: risk-free rate is taken from ^IRX by default and not displayed.
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'AAPL')
        benchmark_symbol = request.form.get('benchmark_symbol', '^GSPC')
        start_date = request.form.get('start_date', '2024-01-01')
        end_date = request.form.get('end_date', '2025-01-01')
        display_period = request.form.get('display_period', 'annual')  # "daily" or "annual"

        # Use default risk-free rate from IRX
        risk_free_rate = functions.get_risk_free_rate(start_date, end_date)

        ticker_data = functions.get_stock(stock_symbol, start_date, end_date)
        benchmark_data = functions.get_benchmark(benchmark_symbol, start_date, end_date)

        beta_value = functions.beta(ticker_data, benchmark_data)
        volatility_value = functions.volatility(ticker_data)  # assumed annualized
        # Use computed beta_value instead of None
        treynor_ratio_value = functions.treynor_ratio(ticker_data, benchmark_data, risk_free_rate, beta_value)
        sharpe_ratio_value = functions.sharpe_ratio(ticker_data, benchmark_data, risk_free_rate)
        alpha_value = functions.alpha(ticker_data, benchmark_data)

        # Convert values for display based on the toggle.
        if display_period == 'annual':
            volatility_disp = volatility_value * 100             # in percentage
            alpha_disp = alpha_value * 252 * 100                  # annual alpha in %
            treynor_disp = treynor_ratio_value                   # already annualized
            sharpe_disp = sharpe_ratio_value                     # already annualized
        else:
            volatility_disp = (volatility_value / np.sqrt(252)) * 100
            alpha_disp = alpha_value * 100
            treynor_disp = treynor_ratio_value / np.sqrt(252)
            sharpe_disp = sharpe_ratio_value / np.sqrt(252)
        
        plot_data = functions.distribution_plot_base64(ticker_data)

        return render_template('index.html',
                               stock_symbol=stock_symbol,
                               benchmark_symbol=benchmark_symbol,
                               start_date=start_date,
                               end_date=end_date,
                               display_period=display_period,
                               beta=beta_value,
                               volatility=volatility_disp,
                               treynor_ratio=treynor_disp,
                               sharpe_ratio=sharpe_disp,
                               alpha=alpha_disp,
                               plot_data=plot_data)
    return render_template('index.html')

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    if request.method == 'POST':
        stock_symbols_str = request.form.get('stock_symbols', 'AAPL,NVDA,TSLA')
        stock_weights_str = request.form.get('stock_weights', '50,30,20')
        benchmark_symbol = request.form.get('benchmark_symbol', '^GSPC')
        start_date = request.form.get('start_date', '2024-01-01')
        end_date = request.form.get('end_date', '2025-01-01')
        display_period = request.form.get('display_period', 'annual')

        # Use default risk-free rate from IRX
        risk_free_rate = functions.get_risk_free_rate(start_date, end_date)
        
        stock_list = [s.strip() for s in stock_symbols_str.split(",")]
        try:
            weights_list = [float(w.strip())/100 for w in stock_weights_str.split(",")]
        except ValueError:
            weights_list = None
        
        portfolio_returns = functions.get_stock(stock_list, start_date, end_date)
        benchmark_returns = functions.get_benchmark(benchmark_symbol, start_date, end_date)
        
        correlation_plot = plot_correlation_matrix(portfolio_returns)
        
        beta = functions.portfolio_beta(portfolio_returns, benchmark_returns, weights_list)
        volatility = functions.portfolio_volatility(portfolio_returns, weights_list)  # annualized volatility
        treynor_ratio = functions.portfolio_treynor_ratio(portfolio_returns, benchmark_returns, risk_free_rate, weights_list)
        sharpe_ratio = functions.portfolio_sharpe_ratio(portfolio_returns, risk_free_rate, weights_list)
        alpha = functions.portfolio_alpha(portfolio_returns, benchmark_returns, weights_list)
        
        if display_period == 'annual':
            volatility_disp = volatility * 100
            alpha_disp = alpha * 252 * 100
            treynor_disp = treynor_ratio
            sharpe_disp = sharpe_ratio
        else:
            volatility_disp = (volatility / np.sqrt(252)) * 100
            alpha_disp = alpha * 100
            treynor_disp = treynor_ratio / np.sqrt(252)
            sharpe_disp = sharpe_ratio / np.sqrt(252)
        
        # Interpret alpha (annualized percentage) to indicate performance relative to the benchmark.
        if alpha_disp > 0:
            alpha_interpretation = f"The portfolio outperformed the benchmark by {alpha_disp:.2f}% annually."
        elif alpha_disp < 0:
            alpha_interpretation = f"The portfolio underperformed the benchmark by {abs(alpha_disp):.2f}% annually."
        else:
            alpha_interpretation = "The portfolio performance was in line with the benchmark."
        
        return render_template('portfolio.html',
                               stock_symbols=stock_symbols_str,
                               stock_weights=stock_weights_str,
                               benchmark_symbol=benchmark_symbol,
                               start_date=start_date,
                               end_date=end_date,
                               display_period=display_period,
                               beta=beta,
                               volatility=volatility_disp,
                               treynor_ratio=treynor_disp,
                               sharpe_ratio=sharpe_disp,
                               alpha=alpha_disp,
                               correlation_plot=correlation_plot,
                               alpha_interpretation=alpha_interpretation)
    return render_template('portfolio.html')

if __name__ == '__main__':
    app.run(debug=True)
