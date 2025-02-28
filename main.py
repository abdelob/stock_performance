import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant
import numpy as np
from flask import Flask, render_template, request, send_file
import functions
import matplotlib
matplotlib.use('Agg') 
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol', 'AAPL')
        benchmark_symbol = request.form.get('benchmark_symbol', '^GSPC')
        start_date = request.form.get('start_date', '2024-01-01')
        end_date = request.form.get('end_date', '2025-01-01')
        display_type = request.form.get('display_type', 'daily')

        # Fetch data
        ticker_data = functions.get_stock(stock_symbol, start_date, end_date)
        benchmark_data = functions.get_benchmark(benchmark_symbol, start_date, end_date)
        risk_free_rate = round(functions.get_risk_free_rate(start_date, end_date), 4)

        # Calculate metrics
        beta_value = functions.beta(ticker_data, benchmark_data)
        volatility_value = round(functions.volatility(ticker_data), 2)
        treynor_ratio_value = functions.treynor_ratio(ticker_data, benchmark_data, risk_free_rate, beta_value)
        sharpe_ratio_value = functions.sharpe_ratio(ticker_data, benchmark_data, risk_free_rate)
        jensen_alpha_value = round(functions.jensen_alpha(ticker_data, benchmark_data), 4)

        # Annualize metrics if display_type is 'annual'
        if display_type == 'annual':
            jensen_alpha_value = round(jensen_alpha_value * 252, 4)
            volatility_value = round(volatility_value * np.sqrt(252), 2)
            risk_free_rate = round(risk_free_rate * 252, 4)

        # Generate the histogram as base64
        plot_data = functions.distribution_plot_base64(ticker_data)

        # Generate CSV file
        csv_filename = f"Stock Data {stock_symbol}.csv"
        functions.data_csv(stock_symbol, start_date, end_date, csv_filename)

        # Render template, now passing display_type
        return render_template(
            'index.html',
            stock_symbol=stock_symbol,
            benchmark_symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
            beta=beta_value,
            volatility=volatility_value,
            treynor_ratio=treynor_ratio_value,
            sharpe_ratio=sharpe_ratio_value,
            jensen_alpha=jensen_alpha_value,
            risk_free_rate=risk_free_rate,
            plot_data=plot_data,
            csv_filename=csv_filename,
            display_type=display_type
        )
    else:
        # On GET, default display type is 'daily'
        return render_template('index.html', display_type='daily')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run()
