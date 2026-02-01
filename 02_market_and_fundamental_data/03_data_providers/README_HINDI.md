# 02 API access to market data

There hain several options to access market data via API use karke Python.

## pandas datareader

Notebook [01_pandas_datareader_demo](01_pandas_datareader_demo.ipynb) presents a few sources built into the pandas library. 
- The `pandas` library enables access to data displayed on websites use karke the read_html function 
- the related `pandas-datareader` library provide karta hai access to the API endpoints ka various data providers through a standard interface 

## yfinance: Yahoo! Finance market aur fundamental data 

Notebook [yfinance_demo](02_yfinance_demo.ipynb) shows how to use yfinance to download a variety ka data from Yahoo! Finance. The library works around the deprecation ka the historical data API by scraping data from the website mein a reliable, efficient way ke saath a Pythonic API.

## LOBSTER tick data

Notebook [03_lobster_itch_data](03_lobster_itch_data.ipynb) demonstrate karta hai the use ka order book data made available by LOBSTER (Limit Order Book System - The Efficient Reconstructor), an [online](https://lobsterdata.com/info/WhatIsLOBSTER.php) limit order book data tool that aims to provide easy-to-use, high-quality limit order book data.

Since 2013 LOBSTER acts as a data provider ke liye the academic community, giving access to reconstructed limit order book data ke liye the entire universe ka NASDAQ traded stocks. More recently, it started offering a commercial service.

## Qandl

Notebook [03_quandl_demo](03_quandl_demo.ipynb) shows how Quandl use karta hai a very straightforward API to make its free aur premium data available. See [documentation](https://www.quandl.com/tools/api) ke liye more details.

## zipline & Qantopian

Notebook [contain karta hai the notebook [zipline_data](05_zipline_data.ipynb) briefly introduces the backtesting library `zipline` that hum will use throughout this book aur show how to access stock price data while running a backtest. ke liye installation please refer to the instructions [here](../../installation).

