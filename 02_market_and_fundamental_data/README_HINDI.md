# Market aur Fundamental Data: Sources aur Techniques

Data hamesha se trading ka ek zaruri driver raha hai, aur traders ne lambe samay se behtar information tak pahunch banakar fayda uthane ki koshish ki hai. Ye koshishein kam se kam un afwahon tak purani hain ki House of Rothschild ne Waterloo mein British jeet ke baare mein kabootaron (pigeons) ke zariye channel ke paar aayi khabar se bonds khareedkar bhaari munafa kamaya tha.

Aaj, tez data access mein nivesh (investment) leading **high-frequency trading** (HFT) firms ke Go West consortium ka roop le chuka hai jo Chicago Mercantile Exchange (CME) ko Tokyo se jodta hai. CME aur New York mein BATS exchanges ke beech round-trip latency (aane-jaane ka samay) theoretical limit eight milliseconds ke kareeb aa gayi hai kyunki traders arbitrage maukon ka fayda uthane ke liye muqabla karte hain. Saath hi, regulators aur exchanges ne 'speed bumps' shuru kiye hain jo trading ko dheema karte hain taaki information tak uneven access ke competition par padne wale bure asar ko seemit (limit) kiya ja sake.

Paramparik taur par (traditionally), investors zyadatar **publicly available market aur fundamental data** par nirbhar rahe hain. Private datasets banane ya khareedne ki koshishein (jaise proprietary surveys ke zariye) seemit thin. Conventional strategies equity fundamentals par focus karti hain aur reported financials par financial models banati hain, inhe shayad industry ya macro data ke saath jodkar earnings per share aur stock prices project kiya jata hai. Vaikalpik roop se (alternatively), wo price aur volume information se compute kiye gaye indicators ka use karke market data se signals nikalne ke liye technical analysis ka labh uthate hain.

**Machine learning (ML) algorithms** market aur fundamental data ko insani niyam aur heuristics ke mukable zyada kushalta se exploit karne ka wada karte hain, khaas taur par jab inhe alternative data (jo agle chapter ka topic hai) ke saath joda jata hai. Hum dikhayenge ki linear models se lekar recurrent neural networks (RNNs) tak ke ML algorithms ko market aur fundamental data par kaise apply karein aur tradeable signals generate karein.

Ye chapter market aur fundamental data sources ko introduce karta hai aur samjhata hai ki wo us environment ko kaise reflect karte hain jisme wo banaye gaye hain. **Trading environment** ki details na sirf market data ke sahi interpretation ke liye mayne rakhti hain balki aapki strategy ke design aur execution, aur realistic backtesting simulations ke implementation ke liye bhi zaruri hain. Hum ye bhi dikhayenge ki Python ka use karke alag-alag sources se trading aur financial statement data ko kaise access karein aur unke saath kaam karein.
 
## Vishay Soochi (Content)

1. [Market data trading environment ko reflect karta hai](#market-data-trading-environment-ko-reflect-karta-hai)
    * [Market microstructure: Trading ki barikiyan (The nuts and bolts of trading)](#market-microstructure-trading-ki-barikiyan)
2. [High-frequency market data ke saath kaam karna](#high-frequency-market-data-ke-saath-kaam-karna)
    * [NASDAQ order book data ke saath kaise kaam karein](#nasdaq-order-book-data-ke-saath-kaise-kaam-karein)
    * [Trades kaise communicate hoti hain: FIX protocol](#trades-kaise-communicate-hoti-hain-fix-protocol)
    * [NASDAQ TotalView-ITCH data feed](#nasdaq-totalview-itch-data-feed)
        - [Code Example: Parsing and normalizing tick data ](#code-example-parsing-and-normalizing-tick-data-)
        - [Additional Resources](#additional-resources)
    * [AlgoSeek minute bars: Equity quote aur trade data](#algoseek-minute-bars-equity-quote-aur-trade-data)
        - [Consolidated feed se minute bars tak](#consolidated-feed-se-minute-bars-tak)
        - [Code Example: AlgoSeek intraday data ko kaise process karein](#code-example-algoseek-intraday-data-ko-kaise-process-karein)
3. [Market Data ke liye API Access](#market-data-ke-liye-api-access)
    * [Pandas ka use karke Remote data access](#pandas-ka-use-karke-remote-data-access)
    * [Code Examples](#code-examples)
    * [Data sources](#data-sources)
    * [Industry News](#industry-news)
4. [Fundamental data ke saath kaise kaam karein](#fundamental-data-ke-saath-kaise-kaam-karein)
    * [Financial statement data](#financial-statement-data)
    * [XBRL markup ka use karke Automated processing](#xbrl-markup-ka-use-karke-automated-processing)
    * [Code Example: Fundamental data time series banana](#code-example-fundamental-data-time-series-banana)
    * [Anya fundamental data sources](#anya-fundamental-data-sources)
5. [Pandas ke saath Efficient data storage](#pandas-ke-saath-efficient-data-storage)
    * [Code Example](#code-example-1)
 
## Market data trading environment ko reflect karta hai

Market data is baat ka product hai ki traders kisi financial instrument ke liye orders kaise place karte hain - seedhe ya intermediaries ke zariye un dher saare marketplaces mein se kisi par, aur unhe kaise process kiya jata hai aur demand aur supuly ko match karke prices kaise set ki jati hain. Natijan, data trading venues ke institutional environment ko reflect karta hai, jisme wo rules aur regulations shamil hain jo orders, trade execution, aur price formation ko govern karte hain. Global overview ke liye [Harris](https://global.oup.com/ushe/product/trading-and-exchanges-9780195144703?cc=us&lang=en&) (2003) dekhein aur US market par details ke liye [Jones](https://www0.gsb.columbia.edu/faculty/cjones/papers/2018.08.31%20US%20Equity%20Market%20Data%20Paper.pdf) (2018) dekhein.

Algorithmic traders buy aur sell orders ke flow aur usse banne wale volume aur price statistics ko analyze karne ke liye algorithms (ML sahit) ka use karte hain. Iska maksad trade signals nikalna hota hai jo example ke liye, demand-supply dynamics ya kuch market participants ke behavior ki insights pakad sakein. Ye section un institutional features ko review karta hai jo backtest ke dauran trading strategy ke simulation par asar dalte hain, isse pehle ki hum actual tick data ke saath kaam karna shuru karein jo aise hi ek environment, yani NASDAQ dwara create kiya gaya hai.

### Market microstructure: Trading ki barikiyan

Market microstructure study karta hai ki institutional environment trading process par kaise asar dalta hai aur outcomes ko kaise shape karta hai, jaise price discovery, bid-ask spreads aur quotes, intraday trading behavior, aur transaction costs. Ye financial research ke sabse tezi se badhne wale fields mein se ek hai, jise algorithmic aur electronic trading ke rapid development ne aage badhaya hai.

Aaj, hedge funds in-house analysts ko sponsor karte hain taaki wo tezi se badalte, complex details ko track kar sakein aur best possible market prices par execution ensure kar sakein aur aisi strategies design kar sakein jo market frictions ka fayda uthayein. Ye section trading dwara generate kiye gaye data mein utarne se pehle key concepts ka brief overview deta hai, jaise alag-alag market places aur order types.

### High-frequency market data ke saath kaam karna

Market data ki do categories US exchanges par listed hazaron companies ko cover karti hain jo Reg NMS ke tehat trade ki jati hain: Consolidated feed har trading venue se trade aur quote data combine karta hai, jabki har individual exchange additional activity information ke saath proprietary products offer karta hai.

Is section mein, hum pehle NASDAQ dwara diya gaya proprietary order flow data prastut karenge jo orders, trades, aur usse banne wale prices ka actual stream darshaata hai jaise wo tick-by-tick basis par hote hain. Phir, hum dikhayenge ki irregular intervals par aane wale is continuous stream of data ko fixed duration ke bars mein kaise regularize karein. Aakhir mein, hum AlgoSeek ke equity minute bar data ko introduce karenge jisme consolidated trade aur quote information hoti hai. Har case mein, hum illustrate karenge ki Python ka use karke data ke saath kaise kaam karein taaki aap apni trading strategy ke liye in sources ka labh utha sakein.

### NASDAQ order book data ke saath kaise kaam karein

Market data ka main source order book hai, jo din bhar real-time mein update hoti rehti hai taaki trading ki saari activity dikhayi de sake. Exchanges aamtaur par ye data real-time service ke roop mein fee lekar dete hain lekin kuch purana (historical) data free mein bhi de sakte hain.

United States mein, stock markets teen tiers mein quotes dete hain, yani Level I, II aur III jo increasingly granular (barik) information aur capabilities dete hain:
- Level I: real-time bid- aur ask-price information, jaisa ki dher saare online sources se available hota hai
- Level II: specific market makers dwara bid aur ask prices ke baare mein information add karta hai, saath hi kisi equity ki liquidity ki behtar insight ke liye haal hi ke transactions ka size aur time bhi deta hai.
- Level III: quotes enter ya change karne, orders execute karne, aur trades confirm karne ki kshamta add karta hai aur ye sirf market makers aur exchange member firms ke liye available hai. Level III quotes tak access registered brokers ko best execution requirements pura karne ki suvidha deta hai.

Trading activity un dher saare messages mein dikhti hai jo market participants orders ke liye bhejte hain. Ye messages aamtaur par electronic Financial Information eXchange (FIX) communications protocol (jo securities transactions aur market data ke real-time exchange ke liye hota hai) ya phir exchange ke apne native protocol ke hisab se hote hain.

### Trades kaise communicate hoti hain: FIX protocol

Trading activity un dher saare messages mein dikhti hai jo market participants trade orders ke liye bhejte hain. Ye messages aamtaur par electronic Financial Information eXchange (FIX) communications protocol (jo securities transactions aur market data ke real-time exchange ke liye hota hai) ya phir exchange ke apne native protocol ke hisab se hote hain.

### NASDAQ TotalView-ITCH data feed

Halaanki FIX ka market share bahut bada hai, exchanges apne native protocols bhi offer karte hain. Nasdaq ek TotalView ITCH direct data-feed protocol deta hai jo subscribers ko equity instruments ke individual orders ko track karne ki suvidha deta hai - order lagne se lekar execute hone ya cancel hone tak.

#### Code Example: Parsing aur normalizing tick data

- Folder [NASDAQ TotalView ITCH Order Book](01_NASDAQ_TotalView-ITCH_Order_Book) mein ye code shamil hai:
    - NASDAQ Total View sample tick data download karne ke liye,
    - binary source data se messages parse karne ke liye,
    - kisi diye gaye stock ke liye order book reconstruct karne ke liye,
    - order flow data ko visualize karne ke liye,
    - tick data ko normalize karne ke liye.
- Binary Data services: `struct` [module](https://docs.python.org/3/library/struct.html)

### AlgoSeek minute bars: Equity quote aur trade data

AlgoSeek historical intraday data us quality par provide karta hai jo pehle sirf institutional investors ke liye available thi. AlgoSeek Equity bars user-friendly format mein ek bahut hi detailed intraday quote aur trade data dete hain jiska maksad intraday ML-driven strategies ko design aur backtest karna aasan banana hai. Jaisa ki hum dekhenge, data mein na sirf OHLCV information shamil hai balki bid-ask spread aur up aur down price moves wale ticks ki sankhya ki jaankari bhi shamil hai.
AlgoSeek ne demonstration purpose ke liye 2013-2017 ke liye NASDAQ 100 stocks ke minute bar data ke samples provide karne ki kripa ki hai aur is data ka ek subset is book ke readers ke liye available karayega.

#### Consolidated feed se minute bars tak

AlgoSeek minute bars Securities Information Processor (SIP) dwara diye gaye data par based hain jo is section ki shuruwat mein bataye gaye consolidated feed ko manage karta hai. Aap documentation https://www.algoseek.com/data-drive.html par dekh sakte hain.

Quote aur trade data fields
Minute bar data mein 54 fields tak hote hain. Bar ke open, high, low, aur close elements ke liye aath fields hain:
- Bar aur corresponding trade ke liye timestamp
- Prevailing bid-ask quote aur relevant trade ke liye price aur size

Bar period ke liye volume information ke saath 14 data points bhi hain:
- Shares ki sankhya aur corresponding trades
- Trade volumes jo bid par ya usse niche, bid quote aur midpoint ke beech, midpoint par, midpoint aur ask quote ke beech, aur ask par ya usse upar huye, saath hi crosses ke liye bhi
- Up- ya downticks ke saath trade huye shares ki sankhya, yani jab price badha ya gira, saath hi jab price change nahi hua, price movement ki pichli direction ke hisab se differentiated

#### Code Example: AlgoSeek intraday data ko kaise process karein

Directory [algoseek_intraday](02_algoseek_intraday) mein AlgoSeek se sample data download karne ke liye instructions hain. 

## Market Data ke liye API Access

Python ka use karke API ke zariye market data access karne ke liye kai options hain. Is chapter mein, hum pehle [`pandas`](https://pandas.pydata.org/) library mein built kuch sources prastut karenge. Phir hum trading platform [Quantopian](https://www.quantopian.com/posts), data provider [Quandl](https://www.quandl.com/) (12/2018 mein NASDAQ dwara acquired) aur backtesting library [`zipline`](https://github.com/quantopian/zipline) (jise hum book mein baad mein use karenge) ko briefly introduce karenge, aur tarah-tarah ke market data access karne ke liye kai additional options list karenge. Directory [data_providers](03_data_providers) mein kai notebooks hain jo in options ka upyog dikhate hain.

### Pandas ka use karke Remote data access

- read_html [docs](https://pandas.pydata.org/pandas-docs/stable/)
- [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) se S&P 500 constituents
- `pandas-datareader`[docs](https://pandas-datareader.readthedocs.io/en/latest/index.html)

### Code Examples

Folder [data providers](03_data_providers) mein alag-alag data providers use karne ke examples hain.
1. [pandas DataReader](03_data_providers/01_pandas_datareader_demo.ipynb) ka use karke Remote data access
2. [yfinance](03_data_providers/02_yfinance_demo.ipynb) ke saath market aur fundamental data Download karna
3. [LOBSTER](03_data_providers/03_lobster_itch_data.ipynb) se Limit Order Tick Data Parse karna
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

## Fundamental data ke saath kaise kaam karein

Fundamental data un economic drivers se juda hai jo securities ki value determine karte hain. Data ka nature asset class par nirbhar karta hai:
- Equities aur corporate credit ke liye, isme corporate financials ke saath-saath industry aur economy-wide data shamil hota hai.
- Government bonds ke liye, isme international macro-data aur foreign exchange shamil hota hai.
- Commodities ke liye, isme asset-specific supply-and-demand determinants shamil hote hain, jaise fasalon ke liye mausam ka data.

Hum US ke liye equity fundamentals par focus karenge, jahan data access karna aasan hai. Duniya bhar mein karib 13,000+ public companies hain jo 2 million pages ki annual reports aur 30,000+ hours ki earnings calls generate karti hain. Algorithmic trading mein, fundamental data aur is data se banaye gaye features ka use trade signals nikalne ke liye seedhe kiya ja sakta hai, example ke liye value indicators ke roop mein, aur ye predictive models (ML models sahit) ke liye ek zaruri input hote hain.

### Financial statement data

Securities and Exchange Commission (SEC) sabhi US issuers, yani listed companies aur securities (mutual funds sahit), ke liye zaruri karta hai ki wo teen quarterly financial statements (Form 10-Q) aur ek annual report (Form 10-K) file karein. Iske alawa aur bhi kai regulatory filing requirements hoti hain.

1990s ki shuruwat se, SEC ne apne Electronic Data Gathering, Analysis, and Retrieval (EDGAR) system ke zariye in filings ko available karaya hai. Ye equity aur anya securities (jaise corporate credit) ke fundamental analysis ke liye primary data source hain, jahan value business prospects aur issuer ki financial health par nirbhar karti hai.

### XBRL markup ka use karke Automated processing

Jab se SEC ne XBRL introduce kiya hai, regulatory filings ka automated analysis bahut aasan ho gaya hai. XBRL business reports ke electronic representation aur exchange ke liye ek free, open, aur global standard hai. XBRL XML par based hai; ye [taxonomies](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) par nirbhar karta hai jo report ke elements ka matlab define karti hain aur unhe tags se map karti hain jo electronic version mein corresponding information ko highlight karte hain. Aisi hi ek taxonomy US Generally Accepted Accounting Principles (GAAP) ko represent karti hai.

SEC ne 2005 mein accounting scandals ke jawab mein voluntary XBRL filings shuru ki thi, aur 2009 se sabhi filers ke liye is format ko zaruri kar diya. Ab wo dusri regulatory filings ke liye bhi mandatory coverage bada raha hai. SEC ek website maintain karta hai jo current taxonomies ki list deti hai, jo alag-alag filings ke content ko shape karti hain aur specific items extract karne ke liye use ki ja sakti hain.

SEC ko report kiye gaye fundamental data ko track aur access karne ke kai tarike hain:
- [EDGAR Public Dissemination Service]((https://www.sec.gov/oit/announcement/public-dissemination-service-system-contact.html)) (PDS) ke hisse ke roop mein, accepted filings ki electronic feeds fee dekar available hain.
- SEC har 10 minute mein [RSS feeds](https://www.sec.gov/structureddata/rss-feeds-submitted-filings) update karta hai, jo structured disclosure submissions list karti hain.
- Automated processing ke liye FTP ke zariye sabhi filings retrieve karne ke liye public [index files](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm) hain.
- Financial statement (aur notes) datasets mein sabhi financial statements aur accompanying notes se parsed XBRL data hota hai.

SEC log files bhi publish karta hai jisme EDGAR filings ke liye [internet search traffic](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) hota hai (SEC.gov ke zariye), halaanki isme cheh mahine ki deri (delay) hoti hai.

### Code Example: Fundamental data time series banana

[Financial Statement and Notes](https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html) datasets mein data ka scope numeric data hota hai jo primary financial statements (Balance sheet, income statement, cash flows, changes in equity, aur comprehensive income) aur un statements ke footnotes se extract kiya jata hai. Data 2009 se available hai.

Folder [04_sec_edgar](04_sec_edgar) mein notebook [edgar_xbrl](04_sec_edgar/edgar_xbrl.ipynb) hai jo EDGAR data ko XBRL format mein download aur parse karne ke liye hai, aur financial statement aur price data ko combine karke P/E ratio jaise fundamental metrics create karta hai.

### Anya fundamental data sources

- [Compilation of macro resources by the Yale Law School](https://library.law.yale.edu/news/75-sources-economic-data-statistics-reports-and-commentary)
- [Capital IQ](www.capitaliq.com)
- [Compustat](www.compustat.com)
- [MSCI Barra](www.mscibarra.com)
- [Northfield Information Services](www.northinfo.com)
- [Quantitative Services Group](www.qsg.com)

## Pandas ke saath Efficient data storage

Hum is book mein kai alag-alag data sets use karenge, aur efficiency aur performance ke liye main formats ko compare karna faydemand hai. Khaas taur par, hum inki tulna karte hain:

- CSV: Comma-separated, standard flat text file format.
- HDF5: Hierarchical data format, jo shuru mein National Center for Supercomputing mein develop kiya gaya tha, numerical data ke liye ek fast aur scalable storage format hai, jo PyTables library ka use karke pandas mein available hai.
- Parquet: Ek binary, columnar storage format jo Apache Hadoop ecosystem ka hissa hai, efficient data compression aur encoding deta hai aur ise Cloudera aur Twitter dwara develop kiya gaya hai. Ye `pyarrow` library ke zariye pandas ke liye available hai, jise pandas ke original author Wes McKinney lead karte hain.

### Code Example

Directory [05_storage_benchmark](05_storage_benchmark) mein notebook [storage_benchmark](05_storage_benchmark/storage_benchmark.ipynb) upar di gayi libraries ki performance ko compare karta hai.
