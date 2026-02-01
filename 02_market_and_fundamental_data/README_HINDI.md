# Market & Fundamental Data: Sources aur Techniques

Data has always been an essential driver ka trading, aur traders have long made efforts to gain an advantage from access to superior information. These efforts date back at least to the rumors that the House ka Rothschild benefited handsomely from bond purchases upon advance news about the British victory at Waterloo carried by pigeons across the channel.

Today, investments mein faster data access take the shape ka the Go West consortium ka leading **high-frequency trading** (HFT) firms that connects the Chicago Mercantile Exchange (CME) ke saath Tokyo. The round-trip latency between the CME aur the BATS exchanges mein New York has dropped to close to the theoretical limit ka eight milliseconds as traders compete to exploit arbitrage opportunities. At the same time, regulators aur exchanges have started to introduce speed bumps that slow down trading to limit the adverse effects on competition ka uneven access to information.

Traditionally, investors mostly relied on **publicly available market aur fundamental data**.  Efforts to create or acquire private datasets, ke liye example through proprietary surveys, were limited. Conventional strategies focus on equity fundamentals aur build financial models on reported financials, possibly combined ke saath industry or macro data to project earnings per share aur stock prices. Alternatively, they leverage technical analysis to extract signals from market data use karke indicators computed from price aur volume information.

**Machine learning (ML) algorithms** promise to exploit market aur fundamental data more efficiently than human-defined rules aur heuristics, mein particular when combined ke saath alternative data, the topic ka the next chapter. hum will illustrate how to apply ML algorithms ranging from linear models to recurrent neural networks (RNNs) to market aur fundamental data aur generate tradeable signals.

Yeh chapter introduces market aur fundamental data sources aur explains how they reflect the environment mein which they hain created. The details ka the **trading environment** matter not only ke liye the proper interpretation ka market data but also ke liye the design aur execution ka your strategy aur the implementation ka realistic backtesting simulations. hum also illustrate how to access aur work ke saath trading aur financial statement data from various sources use karke Python. 
 
## Vishay-suchi (Content)

1. [Market data reflects the trading environment](#market-data-reflects-the-trading-environment)
    * [Market microstructure: The nuts and bolts of trading](#market-microstructure-the-nuts-and-bolts-of-trading)
2. [Working ke saath high-frequency market data](#working-ke saath-high-frequency-market-data)
    * [How to work with NASDAQ order book data](#how-to-work-with-nasdaq-order-book-data)
    * [How trades are communicated: The FIX protocol](#how-trades-are-communicated-the-fix-protocol)
    * [The NASDAQ TotalView-ITCH data feed](#the-nasdaq-totalview-itch-data-feed)
        - [Code Example: Parsing and normalizing tick data ](#code-example-parsing-and-normalizing-tick-data-)
        - [Additional Resources](#additional-resources)
    * [AlgoSeek minute bars: Equity quote and trade data](#algoseek-minute-bars-equity-quote-and-trade-data)
        - [From the consolidated feed to minute bars](#from-the-consolidated-feed-to-minute-bars)
        - [Code Example: How to process AlgoSeek intraday data](#code-example-how-to-process-algoseek-intraday-data)
3. [API Access to Market Data](#api-access-to-market-data)
    * [Remote data access using pandas](#remote-data-access-using-pandas)
    * [Code Examples](#code-examples)
    * [Data sources](#data-sources)
    * [Industry News](#industry-news)
4. [How to work ke saath Fundamental data](#how-to-work-ke saath-fundamental-data)
    * [Financial statement data](#financial-statement-data)
    * [Automated processing using XBRL markup](#automated-processing-using-xbrl-markup)
    * [Code Example: Building a fundamental data time series](#code-example-building-a-fundamental-data-time-series)
    * [Other fundamental data sources](#other-fundamental-data-sources)
5. [Efficient data storage ke saath pandas](#efficient-data-storage-ke saath-pandas)
    * [Code Example](#code-example)
 
## Market data reflects the trading environment

Market data hai the product ka how traders place orders ke liye a financial instrument directly or through intermediaries on one ka the numerous marketplaces aur how they hain processed aur how prices hain set by matching demand aur supply. As a result, the data reflects the institutional environment ka trading venues, including the rules aur regulations that govern orders, trade execution, aur price formation. See [Harris](https://global.oup.com/ushe/product/trading-aur-exchanges-9780195144703?cc=us&lang=en&) (2003) ke liye a global overview aur [Jones](https://www0.gsb.columbia.edu/faculty/cjones/papers/2018.08.31%20US%20Equity%20Market%20Data%20Paper.pdf) (2018) ke liye details on the US market.

Algorithmic traders use algorithms, including ML, to analyze the flow ka buy aur sell orders aur the resulting volume aur price statistics to extract trade signals that capture insights into, ke liye example, demand-supply dynamics or the behavior ka certain market participants. Yeh section reviews institutional features that impact the simulation ka a trading strategy during a backtest before hum start working ke saath actual tick data created by one such environment, namely the NASDAQ.

### Market microstructure: The nuts aur bolts ka trading

Market microstructure studies how the institutional environment affects the trading process aur shapes outcomes like the price discovery, bid-ask spreads aur quotes, intraday trading behavior, aur transaction costs. It hai one ka the fastest-growing fields ka financial research, propelled by the rapid development ka algorithmic aur electronic trading.  

Today, hedge funds sponsor mein-house analysts to track the rapidly evolving, complex details aur ensure execution at the best possible market prices aur design strategies that exploit market frictions. Yeh section provide karta hai a brief overview ka key concepts, namely different market places aur order types, before hum dive into the data generated by trading.

- [Trading aur Exchanges - Market Microstructure ke liye Practitioners](https://global.oup.com/ushe/product/trading-aur-exchanges-9780195144703?cc=us&lang=en&), Larry Harris, Oxford University Press, 2003
- [Understanding the Market ke liye Us Equity Market Data](https://www0.gsb.columbia.edu/faculty/cjones/papers/2018.08.31%20US%20Equity%20Market%20Data%20Paper.pdf), Charles Jones, NYSE, 2018 
- [World Federation ka Exchanges](https://www.world-exchanges.org/our-work/statistics)
- [Econophysics ka Order-driven Markets](https://www.springer.com/gp/book/9788847017658), Abergel et al, 2011
    - Presents the ideas and research from various communities (physicists, economists, mathematicians, financial engineers) on the  modelling and analyzing order-driven markets. Of primary interest in these studies are the mechanisms leading to the statistical regularities of price statistics. Results pertaining to other important issues such as market impact, the profitability of trading strategies, or mathematical models for microstructure effects, are also presented.

## Working ke saath high-frequency market data

Two categories ka market data cover the thousands ka companies listed on US exchanges that hain traded under Reg NMS: The consolidated feed combines trade aur quote data from each trading venue, whereas each individual exchange offers proprietary products ke saath additional activity information ke liye that particular venue.

mein this section, hum will first present proprietary order flow data provided by the NASDAQ that represents the actual stream ka orders, trades, aur resulting prices as they occur on a tick-by-tick basis. Then, hum demonstrate how to regularize this continuous stream ka data that arrives at irregular intervals into bars ka a fixed duration. Finally, hum introduce AlgoSeek’s equity minute bar data that contain karta hai consolidated trade aur quote information. mein each case, hum illustrate how to work ke saath the data use karke Python so you can leverage these sources ke liye your trading strategy.

### How to work ke saath NASDAQ order book data

The primary source ka market data hai the order book, which updates mein real-time throughout the day to reflect all trading activity. Exchanges typically offer this data as a real-time service ke liye a fee but may provide some historical data ke liye free. 

mein the United States, stock markets provide quotes mein three tiers, namely Level I, II aur III that offer increasingly granular information aur capabilities:
- Level I: real-time bid- aur ask-price information, as available from numerous online sources
- Level II: adds information about bid aur ask prices by specific market makers as well as size aur time ka recent transactions ke liye better insights into the liquidity ka a given equity.
- Level III: adds the ability to enter or change quotes, execute orders, aur confirm trades aur hai only available to market makers aur exchange member firms. Access to Level III quotes permits registered brokers to meet best execution requirements.

The trading activity hai reflected mein numerous messages about orders sent by market participants. These messages typically conform to the electronic Financial Information eXchange (FIX) communications protocol ke liye real-time exchange ka securities transactions aur market data or a native exchange protocol. 

- [The Limit Order Book](https://arxiv.org/pdf/1012.0349.pdf)
- [Feature Engineering ke liye Mid-Price Prediction ke saath Deep Learning](https://arxiv.org/abs/1904.05384)
- [Price jump prediction mein Limit Order Book](https://arxiv.org/pdf/1204.1381.pdf)
- [Handling aur visualizing order book data](https://github.com/0b01/recurrent-autoencoder/blob/master/Visualizing%20order%20book.ipynb) by Ricky Han

### How trades hain communicated: The FIX protocol

The trading activity hai reflected mein numerous messages about trade orders sent by market participants. These messages typically conform to the electronic Financial Information eXchange (FIX) communications protocol ke liye real-time exchange ka securities transactions aur market data or a native exchange protocol. 

- [FIX Trading Standards](https://www.fixtrading.org/standards/)
- Python: [Simplefix](https://github.com/da4089/simplefix)
- C++ version: [quickfixengine](http://www.quickfixengine.org/)
- Interactive Brokers [interface](https://www.interactivebrokers.com/en/index.php?f=4988)

### The NASDAQ TotalView-ITCH data feed

While FIX has a dominant large market share, exchanges also offer native protocols. The Nasdaq offers a TotalView ITCH direct data-feed protocol that allows subscribers to track individual orders ke liye equity instruments from placement to execution or cancellation.

- The ITCH [Specifications](http://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf)
- [Sample Files](ftp://emi.nasdaq.com/ITCH/)

#### Code Example: Parsing aur normalizing tick data 

- The folder [NASDAQ TotalView ITCH Order Book](01_NASDAQ_TotalView-ITCH_Order_Book) contain karta hai the notebooks to
    - download NASDAQ Total View sample tick data,
    - parse the messages from the binary source data
    - reconstruct the order book for a given stock
    - visualize order flow data
    - normalize tick data
- Binary Data services: the `struct` [module](https://docs.python.org/3/library/struct.html)
 
#### Additional Resources
 
 - Native exchange protocols [around the world](https://en.wikipedia.org/wiki/List_of_electronic_trading_protocols_
 - [High-frequency trading mein a limit order book](https://www.math.nyu.edu/faculty/avellane/HighFrequencyTrading.pdf), Avellaneda aur Stoikov, Quantitative Finance, Vol. 8, No. 3, April 2008, 217–224
 - [use karke a Simulator to Develop Execution Algorithms](http://www.math.ualberta.ca/~cfrei/PIMS/Almgren5.pdf), Robert Almgren, quantitative brokers, 2016
 - [Backtesting Microstructure Strategies](https://rickyhan.com/jekyll/update/2019/12/22/how-to-simulate-market-microstructure.html), Ricky Han, 2019
- [Optimal High-Frequency Market Making](http://stanford.edu/class/msande448/2018/Final/Reports/gr5.pdf), Fushimi et al, 2018
- [Simulating aur analyzing order book data: The queue-reactive model](https://arxiv.org/pdf/1312.0563.pdf), Huan et al, 2014
- [How does latent liquidity get revealed mein the limit order book?](https://arxiv.org/pdf/1808.09677.pdf), Dall’Amico et al, 2018

### AlgoSeek minute bars: Equity quote aur trade data

AlgoSeek provide karta hai historical intraday data at the quality previously available only to institutional investors. The AlgoSeek Equity bars provide a very detailed intraday quote aur trade data mein a user-friendly format aimed at making it easy to design aur backtest intraday ML-driven strategies. As hum will see, the data includes not only OHLCV information but also information on the bid-ask spread aur the number ka ticks ke saath up aur down price moves, among others.
AlgoSeek has been so kind as to provide samples ka minute bar data ke liye the NASDAQ 100 stocks from 2013-2017 ke liye demonstration purposes aur will make a subset ka this data available to readers ka this book.

#### From the consolidated feed to minute bars

AlgoSeek minute bars hain based on data provided by the Securities Information Processor (SIP) that manages the consolidated feed mentioned at the beginning ka this section. You can find the documentation at https://www.algoseek.com/data-drive.html.

Quote aur trade data fields
The minute bar data contain up to 54 fields. There hain eight fields ke liye the open, high, low, aur close elements ka the bar, namely:
- The timestamp ke liye the bar aur the corresponding trade 
- The price aur the size ke liye the prevailing bid-ask quote aur the relevant trade

There hain also 14 data points ke saath volume information ke liye the bar period:
- The number ka shares aur corresponding trades
- The trade volumes at or below the bid, between the bid quote aur the midpoint, at the midpoint, between the midpoint aur the ask quote, aur at or above the ask, as well as ke liye crosses
- The number ka shares traded ke saath up- or downticks, i.e., when the price rose or fell, as well as when the price did not change, differentiated by the previous direction ka price movement

#### Code Example: How to process AlgoSeek intraday data

The directory [algoseek_intraday](02_algoseek_intraday) contain karta hai instructions on how to download sample data from AlgoSeek. 

- This information will be made available shortly.

## API Access to Market Data

There hain several options to access market data via API use karke Python. mein this chapter, hum first present a few sources built into the [`pandas`](https://pandas.pydata.org/) library. Then hum briefly introduce the trading platform [Quantopian](https://www.quantopian.com/posts), the data provider [Quandl](https://www.quandl.com/) (acquired by NASDAQ mein 12/2018) aur the backtesting library [`zipline`](https://github.com/quantopian/zipline) that hum will use later mein the book, aur list several additional options to access various types ka market data. The directory [data_providers](03_data_providers) contain karta hai several notebooks that illustrate the usage ka these options.

### Remote data access use karke pandas

- read_html [docs](https://pandas.pydata.org/pandas-docs/stable/)
- S&P 500 constituents from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- `pandas-datareader`[docs](https://pandas-datareader.readthedocs.io/en/latest/index.html)

### Code Examples

The folder [data providers](03_data_providers) contain karta hai examples to use various data providers.
1. Remote data access use karke [pandas DataReader](03_data_providers/01_pandas_datareader_demo.ipynb)
2. Downloading market aur fundamental data ke saath [yfinance](03_data_providers/02_yfinance_demo.ipynb)
3. Parsing Limit Order Tick Data from [LOBSTER](03_data_providers/03_lobster_itch_data.ipynb)
4. Quandl [API Demo](03_data_providers/04_quandl_demo.ipynb)
5. Zipline [data access](03_data_providers/05_zipline_data_demo.ipynb)

### Data sources

- Quandl [docs](https://docs.quandl.com/docs) aur Python [API](https://www.quandl.com/tools/python﻿)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Quantopian](https://www.quantopian.com/posts)
- [Zipline](https://zipline.ml4trading.io/﻿)
- [LOBSTER](https://lobsterdata.com/)
- [The Investor Exchange](https://iextrading.com/﻿)
- [IEX Cloud](https://iexcloud.io/) financial data infrastructure
- [Money.net](https://www.money.net/)
- [Trading Economic](https://tradingeconomics.com/)
- [Barchart](https://www.barchart.com/)
- [Alpha Vantage](https://www.alphavantage.co/﻿)
- [Alpha Trading Labs](https://www.alphatradinglabs.com/)
- [Tiingo](https://www.tiingo.com/) stock market tools

### Industry News

- [Bloomberg aur Reuters lose data share to smaller rivals](https://www.ft.com/content/622855dc-2d31-11e8-9b4b-bc4b9f08f381), FT, 2018

## How to work ke saath Fundamental data

Fundamental data pertains to the economic drivers that determine the value ka securities. The nature ka the data depends on the asset class:
- ke liye equities aur corporate credit, it includes corporate financials as well as industry aur economy-wide data.
- ke liye government bonds, it includes international macro-data aur foreign exchange.
- ke liye commodities, it includes asset-specific supply-aur-demand determinants, such as weather data ke liye crops. 

hum will focus on equity fundamentals ke liye the US, where data hai easier to access. There hain some 13,000+ public companies worldwide that generate 2 million pages ka annual reports aur 30,000+ hours ka earnings calls. mein algorithmic trading, fundamental data aur features engineered from this data may be used to derive trading signals directly, ke liye example as value indicators, aur hain an essential input ke liye predictive models, including machine learning models.

### Financial statement data

The Securities aur Exchange Commission (SEC) requires US issuers, that hai, listed companies aur securities, including mutual funds to file three quarterly financial statements (Form 10-Q) aur one annual report (Form 10-K), mein addition to various other regulatory filing requirements.

Since the early 1990s, the SEC made these filings available through its Electronic Data Gathering, Analysis, aur Retrieval (EDGAR) system. They constitute the primary data source ke liye the fundamental analysis ka equity aur other securities, such as corporate credit, where the value depends on the business prospects aur financial health ka the issuer. 

### Automated processing use karke XBRL markup

Automated analysis ka regulatory filings has become much easier since the SEC introduced XBRL, a free, open, aur global standard ke liye the electronic representation aur exchange ka business reports. XBRL hai based on XML; it relies on [taxonomies](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) that define the meaning ka the elements ka a report aur map to tags that highlight the corresponding information mein the electronic version ka the report. One such taxonomy represents the US Generally Accepted Accounting Principles (GAAP).

The SEC introduced voluntary XBRL filings mein 2005 mein response to accounting scandals before requiring this format ke liye all filers since 2009 aur continues to expand the mandatory coverage to other regulatory filings. The SEC maintains a website that lists the current taxonomies that shape the content ka different filings aur can be used to extract specific items.

There hain several avenues to track aur access fundamental data reported to the SEC:
- Ke hisse ke roop mein the [EDGAR Public Dissemination Service]((https://www.sec.gov/oit/announcement/public-dissemination-service-system-contact.html)) (PDS), electronic feeds ka accepted filings hain available ke liye a fee. 
- The SEC updates [RSS feeds](https://www.sec.gov/structureddata/rss-feeds-submitted-filings) every 10 minutes, which list structured disclosure submissions.
- There hain public [index files](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm) ke liye the retrieval ka all filings through FTP ke liye automated processing.
- The financial statement (aur notes) datasets contain parsed XBRL data from all financial statements aur the accompanying notes.

The SEC also publishes log files containing the [internet search traffic](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) ke liye EDGAR filings through SEC.gov, albeit ke saath a six-month delay.

### Code Example: Building a fundamental data time series

The scope ka the data mein the [Financial Statement aur Notes](https://www.sec.gov/dera/data/financial-statement-aur-notes-data-set.html) datasets consists ka numeric data extracted from the primary financial statements (Balance sheet, income statement, cash flows, changes mein equity, aur comprehensive income) aur footnotes on those statements. The data hai available as early as 2009.

The folder [04_sec_edgar](04_sec_edgar) contain karta hai the notebook [edgar_xbrl](04_sec_edgar/edgar_xbrl.ipynb) to download aur parse EDGAR data mein XBRL format, aur create fundamental metrics like the P/E ratio by combining financial statement aur price data.

### Other fundamental data sources

- [Compilation ka macro resources by the Yale Law School](https://library.law.yale.edu/news/75-sources-economic-data-statistics-reports-aur-commentary)
- [Capital IQ](www.capitaliq.com)
- [Compustat](www.compustat.com)
- [MSCI Barra](www.mscibarra.com)
- [Northfield Information Services](www.northinfo.com)
- [Quantitative Services Group](www.qsg.com)

## Efficient data storage ke saath pandas

hum'll be use karke many different data sets mein this book, aur it's worth comparing the main formats ke liye efficiency aur performance. mein particular, hum compare the following:

- CSV: Comma-separated, standard flat text file format.
- HDF5: Hierarchical data format, developed initially at the National Center ke liye Supercomputing, hai a fast aur scalable storage format ke liye numerical data, available mein pandas use karke the PyTables library.
- Parquet: A binary, columnar storage format, part ka the Apache Hadoop ecosystem, that provide karta hai efficient data compression aur encoding aur has been developed by Cloudera aur Twitter. It hai available ke liye pandas through the pyarrow library, led by Wes McKinney, the original author ka pandas.

### Code Example

Notebook [storage_benchmark](05_storage_benchmark/storage_benchmark.ipynb) mein the directory [05_storage_benchmark](05_storage_benchmark) compares the performance ka the preceding libraries.
