# How to process AlgoSeek intraday NASDAQ 100 data

You can download a sammle ka Algoseek's NASDAQ100 Minute Bar data ke saath Trade & Quote information ke liye 2015-2017 from Algoseek's website [here](https://www.algoseek.com/ml4t-book-data.html). Notebook [algoseek_minute_data](1_algoseek_minute_data.ipynb) contain karta hai the code to extract aur combine the data that hum will use mein [Chapter 12](../../12_gradient_boosting_machines) to develop a Gradient Boosting model that predicts one-minute returns ke liye an intraday trading strategy.   

Unzip the directory, rename it `1min_taq`, aur move it into a new `nasdaq100` folder mein the [data](../../data) directory. It contain karta hai around 5GB worth ka NASDAQ 100 minute bar data mein trade-aur-quote format. See [documentation](https://us-equity-market-data-docs.s3.amazonaws.com/algoseek.US.Equity.TAQ.Minute.Bars.pdf) ke liye details on the definition ka the numerous fields.
The following information hai from the Algoseek Trade & Quote Minute Bar data linked above. 

## Trade & Quote Minute Bar Fields

The Quote fields hain based on changes to the NBBO ([National Best Bid Offer](https://www.investopedia.com/terms/n/nbbo.asp)) from the top-ka-book price aur size from  each ka the exchanges.

The enhanced Trade & Quote bar fields include the following fields:
- **Field**: Name ka Field.
- **Q / T**: Field based on Quotes or Trades
- **Type**: Field format
- **No Value**: Value ka field when there hai no value or data. 
  - Note: “Never” means field should always have a value EXCEPT ke liye the first bar ka the day.
- **Description**: Description ka the field.

| id  | Field                   | Q/T  | Type                          |  No Value | Description                                                                                                                                                                                                         |
|:---:|-------------------------|:----:|-------------------------------|:---------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | `Date`                   |      | YYYYMMDD                      | Never     | Trade Date                                                                                                                                                                                                          |
| 2  | `Ticker`                 |      | String                       | Never      | Ticker Symbol                                                                                                                                                                                                       |
| 3  | `TimeBarStart`           |      | HHMM <br>HHMMSS <br>HHMMSSMMM | Never     | ke liye minute bars: HHMM. <br>ke liye second bars: HHMMSS. <br>Examples<br>- One second bar 130302 hai from time greater than 130301 to 130302.<br>- One minute bar 1104 hai from time greater than 1103 to 1104. |
| 4  | `OpenBarTime`            | Q    | HHMMSSMMM                    | Never      | Open Time ka the Bar, ke liye example one minute:<br>11:03:00.000                                                                                                                                                       |
| 5  | `OpenBidPrice`           | Q    | Number                        | Never     | NBBO Bid Price as ka bar Open                                                                                                                                                                                       |
| 6  | `OpenBidSize`            | Q    | Number                        | Never     | Total Size from all Exchanges ke saath<br>OpenBidPrice                                                                                                                                                                  |
| 7  | `OpenAskPrice`           | Q    | Number                        | Never     | NBBO Ask Price as ka bar Open                                                                                                                                                                                       |
| 8  | `OpenAskSize`            | Q    | Number                        | Never     | Total Size from all Exchange ke saath<br>OpenAskPrice                                                                                                                                                                   |
| 9  | `FirstTradeTime`         | T    | HHMMSSMMM                     | Blank     | Time ka first Trade                                                                                                                                                                                                 |
| 10 | `FirstTradePrice`        | T    | Number                        | Blank     | Price ka first Trade                                                                                                                                                                                                |
| 11 | `FirstTradeSize`         | T    | Number                        | Blank     | Number ka shares ka first trade                                                                                                                                                                                     |
| 12 | `HighBidTime`            | Q    | HHMMSSMMM                     | Never     | Time ka highest NBBO Bid Price                                                                                                                                                                                      |
| 13 | `HighBidPrice`           | Q    | Number                        | Never     | Highest NBBO Bid Price                                                                                                                                                                                              |
| 14 | `HighBidSize`            | Q    | Number                        | Never     | Total Size from all Exchanges ke saath HighBidPrice                                                                                                                                                                  |
| 15 | `AskPriceAtHighBidPrice` | Q    | Number                        | Never     | Ask Price at time ka Highest Bid Price                                                                                                                                                                              |
| 16 | `AskSizeAtHighBidPrice`  | Q    | Number                        | Never     | Total Size from all Exchanges ke saath `AskPriceAtHighBidPrice`                                                                                                                                                        |
| 17 | `HighTradeTime`          | T    | HHMMSSMMM                     | Blank     | Time ka Highest Trade                                                                                                                                                                                               |
| 18 | `HighTradePrice`         | T    | Number                        | Blank     | Price ka highest Trade                                                                                                                                                                                              |
| 19 | `HighTradeSize`          | T    | Number                        | Blank     | Number ka shares ka highest trade                                                                                                                                                                                   |
| 20 | `LowBidTime`             | Q    | HHMMSSMMM                     | Never     | Time ka lowest Bid                                                                                                                                                                                                  |
| 21 | `LowBidPrice`            | Q    | Number                        | Never     | Lowest NBBO Bid price ka bar.                                                                                                                                                                                       |
| 22 | `LowBidSize`             | Q    | Number                        | Never     | Total Size from all Exchanges ke saath `LowBidPrice`                                                                                                                                                                   |
| 23 | `AskPriceAtLowBidPrice`  | Q    | Number                        | Never     | Ask Price at lowest Bid price                                                                                                                                                                                       |
| 24  | `AskSizeAtLowBidPrice`  | Q    | Number                        | Never     | Total Size from all Exchanges ke saath `AskPriceAtLowBidPrice`                                                                                                                                                                                       |
| 25  | `LowTradeTime`          | T    | HHMMSSMMM                     | Blank     | Time ka lowest Trade                                                                                                                                                                                                                             |
| 26  | `LowTradePrice`         | T    | Number                        | Blank     | Price ka lowest Trade                                                                                                                                                                                                                            |
| 27  | `LowTradeSize`          | T    | Number                        | Blank     | Number ka shares ka lowest trade                                                                                                                                                                                                                 |
| 28  | `CloseBarTime`          | Q    | HHMMSSMMM                     | Never     | Close Time ka the Bar, ke liye example one minute: 11:03:59.999                                                                                                                                                                                      |
| 29  | `CloseBidPrice`         | Q    | Number                        | Never     | NBBO Bid Price at bar Close                                                                                                                                                                                                                      |
| 30  | `CloseBidSize`          | Q    | Number                        | Never     | Total Size from all Exchange ke saath `CloseBidPrice`                                                                                                                                                                                                |
| 31  | `CloseAskPrice`         | Q    | Number                        | Never     | NBBO Ask Price at bar Close                                                                                                                                                                                                                      |
| 32  | `CloseAskSize`          | Q    | Number                        | Never     | Total Size from all Exchange ke saath `CloseAskPrice`                                                                                                                                                                                                |
| 33  | `LastTradeTime`         | T    | HHMMSSMMM                     | Blank     | Time ka last Trade                                                                                                                                                                                                                               |
| 34  | `LastTradePrice`        | T    | Number                        | Blank     | Price ka last Trade                                                                                                                                                                                                                              |
| 35  | `LastTradeSize`         | T    | Number                        | Blank     | Number ka shares ka last trade                                                                                                                                                                                                                   |
| 36  | `MinSpread`             | Q    | Number                        | Never     | Minimum Bid-Ask spread size. Yeh may be 0 if the market was crossed during the bar.<br/>If negative spread due to back quote, make it 0.                                                                                                            |
| 37  | `MaxSpread`             | Q    | Number                        | Never     | Maximum Bid-Ask spread mein bar                                                                                                                                                                                                                    |
| 38  | `CancelSize`            | T    | Number                        | 0         | Total shares canceled. Default=blank                                                                                                                                                                                                             |
| 39  | `VolumeWeightPrice`     | T    | Number                        | Blank     | Trade Volume weighted average price <br>Sum((`Trade1Shares`*`Price`)+(`Trade2Shares`*`Price`)+…)/`TotalShares`. <br>Note: Blank if no trades.                                                                                                        |
| 40  | `NBBOQuoteCount`        | Q    | Number                        | 0         | Number ka Bid aur Ask NNBO quotes during bar period.                                                                                                                                                                                             |
| 41  | `TradeAtBid`            | Q,T  | Number                        | 0         | Sum ka trade volume that occurred at or below the bid (a trade reported/printed late can be below current bid).                                                                                                                                  |
| 42  | `TradeAtBidMid`         | Q,T  | Number                        | 0         | Sum ka trade volume that occurred between the bid aur the mid-point:<br/>(Trade Price > NBBO Bid ) & (Trade Price < NBBO Mid )                                                                                                                       |
| 43  | `TradeAtMid`            | Q,T  | Number                        | 0         | Sum ka trade volume that occurred at mid.<br/>TradePrice = NBBO MidPoint                                                                                                                                                                             |
| 44  | `TradeAtMidAsk`         | Q,T  | Number                        | 0         | Sum ka ask volume that occurred between the mid aur ask:<br/>(Trade Price > NBBO Mid) & (Trade Price < NBBO Ask)                                                                                                                                     |
| 45  | `TradeAtAsk`            | Q,T  | Number                        | 0         | Sum ka trade volume that occurred at or above the Ask.                                                                                                                                                                                           |
| 46  | `TradeAtCrossOrLocked`  | Q,T  | Number                        | 0         | Sum ka trade volume ke liye bar when national best bid/offer hai locked or crossed. <br>Locked hai Bid = Ask <br>Crossed hai Bid > Ask                                                                                                                  |
| 47  | `Volume`                | T    | Number                        | 0         | Total number ka shares traded                                                                                                                                                                                                                    |
| 48  | `TotalTrades`           | T    | Number                        | 0         | Total number ka trades                                                                                                                                                                                                                           |
| 49  | `FinraVolume`           | T    | Number                        | 0         | Number ka shares traded that hain reported by FINRA. <br/>Trades reported by FINRA hain from broker-dealer internalization, dark pools, Over-The-Counter, etc.<br/>FINRA trades represent volume that hai hidden or not public available to trade.         |
| 50  | `UptickVolume`          | T    | Integer                       | 0         | Total number ka shares traded ke saath upticks during bar.<br/>An uptick = ( trade price > last trade price )                                                                                                                                                                                                                               |
| 51  | `DowntickVolume`        | T    | Integer                       | 0         | Total number ka shares traded ke saath downticks during bar.<br/>A downtick = ( trade price < last trade price )                                                                                                                                                                                                                            |
| 52  | `RepeatUptickVolume`    | T    | Integer                       | 0         | Total number ka shares where trade price hai the same (repeated) aur last price change was up during bar. <br/>Repeat uptick = ( trade price == last trade price ) & (last tick direction == up )                                                                                                                                         |
| 53  | `RepeatDowntickVolume`  | T    | Integer                       | 0         | Total number ka shares where trade price hai the same (repeated) aur last price change was down during bar. <br/>Repeat downtick = ( trade price == last trade price ) & (last tick direction == down )                                                                                                                                   |
| 54  | `UnknownVolume`         | T    | Integer                       | 0         | When the first trade ka the day takes place, the tick direction hai “unknown” as there hai no previous Trade to compare it to.<br/>This field hai the volume ka the first trade after 4am aur acts as an initiation value ke liye the tick volume directions.<br/>mein future this bar will be renamed to `UnkownTickDirectionVolume` .  |

### Notes

**Empty Fields**

An empty field has no value aur hai “Blank” , ke liye example FirstTradeTime aur there hain no trades during the bar period. 
The field `Volume` measuring total number ka shares traded mein bar will be `0` if there hain no Trades (see `No Value` column above ke liye each field).

**No Bid/Ask/Trade OHLC**

During a bar timeframe there may not be a change mein the NBBO or an actual Trade. 
Udaharan ke liye, there can be a bar ke saath OHLC Bid/Ask but no Trade OHLC.

**Single Event**

ke liye bars ke saath only one trade, one NBBO bid or one NBBO ask then Open/High/Low/Close price,size andtime will be the same.

**`AskPriceAtHighBidPrice`, `AskSizeAtHighBidPrice`, `AskPriceAtLowBidPrice`, `AskSizeAtLowBidPrice` Fields** 

To provide consistent Bid/Ask prices at a point mein time while showing the low/high Bid/Ask ke liye the bar, AlgoSeek use karta hai the low/high `Bid` aur the corresponding `Ask` at that price.

## FAQ

**Why hain Trade Prices often inside the Bid Price to Ask Price range?**

The Low/High Bid/Ask hai the low aur high NBBO price ke liye the bar range. 
Very often a Trade may not occur at these prices as the price may only last a few seconds or executions hain being crossed at mid-point due to hidden order types that execute at mid-point or as price improvement over current `Bid`/`Ask`.

**How to get exchange tradable shares?** 

To get the exchange tradable volume mein a bar subtract `Volume` from `FinraVolume`. 
- `Volume` hai the total number ka shares traded. 
- ``FinraVolume`` hai the total number ka shares traded that hain reported as executions by FINRA. 

When a trade hai done that hai off the listed exchanges, it must be reported to FINRA by the brokerage firm or dark pool. Examples include: 
- internal crosses by broker dealer
- over-the-counter block trades, aur
- dark pool executions.
