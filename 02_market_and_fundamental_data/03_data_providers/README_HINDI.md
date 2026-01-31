# 02 Market data ke liye API access

Python ka use karke API ke zariye market data access karne ke liye kai options hain.

## pandas datareader

Notebook [01_pandas_datareader_demo](01_pandas_datareader_demo.ipynb) (iska Hindi version `01_pandas_datareader_demo_HINDI.ipynb` bhi hai) pandas library mein built thode sources prastut karta hai.
- `pandas` library read_html function ka use karke websites par dikhaye gaye data ko access karne mein madad karti hai
- related `pandas-datareader` library ek standard interface ke zariye alag-alag data providers ke API endpoints tak pahunchne ki suvidha deti hai

## yfinance: Yahoo! Finance market aur fundamental data 

Notebook [yfinance_demo](02_yfinance_demo.ipynb) dikhata hai ki Yahoo! Finance se tarah-tarah ka data download karne ke liye yfinance ka use kaise karein. Ye library historical data API ke band (depreciation) ho jane ke bawajood website se data ko Pythonic API ke saath reliable aur efficient tarike se scrape karke kaam karti hai.

## LOBSTER tick data

Notebook [03_lobster_itch_data](03_lobster_itch_data.ipynb) LOBSTER (Limit Order Book System - The Efficient Reconstructor) dwara available karaye gaye order book data ka use dikhata hai. LOBSTER ek [online](https://lobsterdata.com/info/WhatIsLOBSTER.php) limit order book data tool hai jiska maksad aasan aur high-quality limit order book data dena hai.

2013 se LOBSTER academic community ke liye data provider ke roop mein kaam kar raha hai, aur NASDAQ par trade hone wake sabhi stocks ke liye reconstructed limit order book data deta hai. Haal hi mein, isne commercial service dena bhi shuru kiya hai.

## Quandl

Notebook [03_quandl_demo](03_quandl_demo.ipynb) dikhata hai ki Quandl apne free aur premium data ko available karane ke liye bahut hi seedha (straightforward) API use karta hai. Zyada jaankari ke liye [documentation](https://www.quandl.com/tools/api) dekhein.

## zipline & Quantopian

Notebook [zipline_data](05_zipline_data_demo.ipynb) backtesting library `zipline` ka brief introduction deta hai jiska use hum is book mein karenge, aur dikhata hai ki backtest chalate waqt stock price data ko kaise access karein. Installation ke liye kripya instructions [yahan](../../installation) dekhein.
