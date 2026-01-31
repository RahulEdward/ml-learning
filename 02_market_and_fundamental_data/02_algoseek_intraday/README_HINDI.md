# AlgoSeek intraday NASDAQ 100 data ko kaise process karein

Aap 2015-2017 ke liye Trade & Quote information ke saath Algoseek ke NASDAQ100 Minute Bar data ka sample Algoseek ki website se [yahan](https://www.algoseek.com/ml4t-book-data.html) download kar sakte hain. Notebook [algoseek_minute_data](1_algoseek_minute_data.ipynb) (iska Hindi version `algoseek_minute_data_HINDI.ipynb` bhi available hai) mein wo code hai jisse hum data extract aur combine karenge. Is data ka use hum [Chapter 12](../../12_gradient_boosting_machines) mein ek Gradient Boosting model banane ke liye karenge jo intraday trading strategy ke liye one-minute returns predict karega.

Directory ko unzip karein, iska naam badal kar `1min_taq` rakhein, aur ise [data](../../data) directory mein ek naye `nasdaq100` folder mein move karein. Isme trade-and-quote format mein lagbhag 5GB ka NASDAQ 100 minute bar data hai. Fields ki definition ke liye details [documentation](https://us-equity-market-data-docs.s3.amazonaws.com/algoseek.US.Equity.TAQ.Minute.Bars.pdf) mein dekhein.
Niche di gayi jaankari upar link kiye gaye Algoseek Trade & Quote Minute Bar data se li gayi hai.

## Trade & Quote Minute Bar Fields

Quote fields har exchange se top-of-book price aur size se NBBO ([National Best Bid Offer](https://www.investopedia.com/terms/n/nbbo.asp)) mein hone wale badlav par based hain.

Enhanced Trade & Quote bar fields mein niche diye gaye fields shamil hain:
- **Field**: Field ka Naam.
- **Q / T**: Field Quotes par based hai ya Trades par
- **Type**: Field ka format
- **No Value**: Field ki value jab koi data na ho. 
  - Note: “Never” ka matlab hai field mein hamesha value honi chahiye SIVAYE din ke pehle bar ke.
- **Description**: Field ka vivaran (description).

| id  | Field                   | Q/T  | Type                          |  No Value | Description (Vivaran)                                                                                                                                                                                                         |
|:---:|-------------------------|:----:|-------------------------------|:---------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | `Date`                   |      | YYYYMMDD                      | Never     | Trade Date (Tareekh)                                                                                                                                                                                                          |
| 2  | `Ticker`                 |      | String                       | Never      | Ticker Symbol                                                                                                                                                                                                       |
| 3  | `TimeBarStart`           |      | HHMM <br>HHMMSS <br>HHMMSSMMM | Never     | Minute bars ke liye: HHMM. <br>Second bars ke liye: HHMMSS. <br>Examples<br>- One second bar 130302 time 130301 se 130302 tak hai.<br>- One minute bar 1104 time 1103 se 1104 tak hai. |
| 4  | `OpenBarTime`            | Q    | HHMMSSMMM                    | Never      | Bar ka Open Time, example 1 minute ke liye:<br>11:03:00.000                                                                                                                                                       |
| 5  | `OpenBidPrice`           | Q    | Number                        | Never     | Bar Open hone par NBBO Bid Price                                                                                                                                                                                       |
| 6  | `OpenBidSize`            | Q    | Number                        | Never     | Sabhi Exchanges se Total Size jahan `OpenBidPrice` hai                                                                                                                                                                  |
| 7  | `OpenAskPrice`           | Q    | Number                        | Never     | Bar Open hone par NBBO Ask Price                                                                                                                                                                                       |
| 8  | `OpenAskSize`            | Q    | Number                        | Never     | Sabhi Exchanges se Total Size jahan `OpenAskPrice` hai                                                                                                                                                                   |
| 9  | `FirstTradeTime`         | T    | HHMMSSMMM                     | Blank     | Pehli Trade ka Time                                                                                                                                                                                                 |
| 10 | `FirstTradePrice`        | T    | Number                        | Blank     | Pehli Trade ka Price                                                                                                                                                                                                |
| 11 | `FirstTradeSize`         | T    | Number                        | Blank     | Pehli trade ke shares ki sankhya                                                                                                                                                                                     |
| 12 | `HighBidTime`            | Q    | HHMMSSMMM                     | Never     | Highest NBBO Bid Price ka Time                                                                                                                                                                                      |
| 13 | `HighBidPrice`           | Q    | Number                        | Never     | Highest NBBO Bid Price                                                                                                                                                                                              |
| 14 | `HighBidSize`            | Q    | Number                        | Never     | Sabhi Exchanges se Total Size jahan `HighBidPrice` hai                                                                                                                                                                  |
| 15 | `AskPriceAtHighBidPrice` | Q    | Number                        | Never     | Highest Bid Price ke waqt Ask Price                                                                                                                                                                              |
| 16 | `AskSizeAtHighBidPrice`  | Q    | Number                        | Never     | Sabhi Exchanges se Total Size jahan `AskPriceAtHighBidPrice` hai                                                                                                                                                        |
| 17 | `HighTradeTime`          | T    | HHMMSSMMM                     | Blank     | Highest Trade ka Time                                                                                                                                                                                               |
| 18 | `HighTradePrice`         | T    | Number                        | Blank     | Highest Trade ka Price                                                                                                                                                                                              |
| 19 | `HighTradeSize`          | T    | Number                        | Blank     | Highest Trade ke shares ki sankhya                                                                                                                                                                                   |
| 20 | `LowBidTime`             | Q    | HHMMSSMMM                     | Never     | Lowest Bid ka Time                                                                                                                                                                                                  |
| 21 | `LowBidPrice`            | Q    | Number                        | Never     | Bar ka Lowest NBBO Bid price                                                                                                                                                                                       |
| 22 | `LowBidSize`             | Q    | Number                        | Never     | Sabhi Exchanges se Total Size jahan `LowBidPrice` hai                                                                                                                                                                   |
| 23 | `AskPriceAtLowBidPrice`  | Q    | Number                        | Never     | Lowest Bid price par Ask Price                                                                                                                                                                                       |
| 24  | `AskSizeAtLowBidPrice`  | Q    | Number                        | Never     | Sabhi Exchanges se Total Size jahan `AskPriceAtLowBidPrice` hai                                                                                                                                                                                       |
| 25  | `LowTradeTime`          | T    | HHMMSSMMM                     | Blank     | Lowest Trade ka Time                                                                                                                                                                                                                             |
| 26  | `LowTradePrice`         | T    | Number                        | Blank     | Lowest Trade ka Price                                                                                                                                                                                                                            |
| 27  | `LowTradeSize`          | T    | Number                        | Blank     | Lowest Trade ke shares ki sankhya                                                                                                                                                                                                                 |
| 28  | `CloseBarTime`          | Q    | HHMMSSMMM                     | Never     | Bar ka Close Time, example 1 minute: 11:03:59.999                                                                                                                                                                                      |
| 29  | `CloseBidPrice`         | Q    | Number                        | Never     | Bar Close par NBBO Bid Price                                                                                                                                                                                                                      |
| 30  | `CloseBidSize`          | Q    | Number                        | Never     | Bar Close par `CloseBidPrice` ke saath Sabhi Exchange se Total Size                                                                                                                                                                                                |
| 31  | `CloseAskPrice`         | Q    | Number                        | Never     | Bar Close par NBBO Ask Price                                                                                                                                                                                                                      |
| 32  | `CloseAskSize`          | Q    | Number                        | Never     | Bar Close par `CloseAskPrice` ke saath Sabhi Exchange se Total Size                                                                                                                                                                                                |
| 33  | `LastTradeTime`         | T    | HHMMSSMMM                     | Blank     | Last Trade ka Time                                                                                                                                                                                                                               |
| 34  | `LastTradePrice`        | T    | Number                        | Blank     | Last Trade ka Price                                                                                                                                                                                                                              |
| 35  | `LastTradeSize`         | T    | Number                        | Blank     | Last trade ke shares ki sankhya                                                                                                                                                                                                                   |
| 36  | `MinSpread`             | Q    | Number                        | Never     | Minimum Bid-Ask spread size. Ye 0 ho sakta hai agar bar ke dauran market cross hui ho.<br/>Agar back quote ki wajah se negative spread hai, to ise 0 manein.                                                                                                            |
| 37  | `MaxSpread`             | Q    | Number                        | Never     | Bar mein Maximum Bid-Ask spread                                                                                                                                                                                                                    |
| 38  | `CancelSize`            | T    | Number                        | 0         | Total cancel huye shares. Default=blank                                                                                                                                                                                                             |
| 39  | `VolumeWeightPrice`     | T    | Number                        | Blank     | Trade Volume weighted average price <br>Sum((`Trade1Shares`*`Price`)+(`Trade2Shares`*`Price`)+…)/`TotalShares`. <br>Note: Agar trades nahi hain to Blank.                                                                                                        |
| 40  | `NBBOQuoteCount`        | Q    | Number                        | 0         | Bar period ke dauran Bid aur Ask NNBO quotes ki sankhya.                                                                                                                                                                                             |
| 41  | `TradeAtBid`            | Q,T  | Number                        | 0         | Us trade volume ka Sum jo bid par ya usse niche hua (koi late report/print hua trade current bid se niche ho sakta hai).                                                                                                                                  |
| 42  | `TradeAtBidMid`         | Q,T  | Number                        | 0         | Us trade volume ka Sum jo bid aur mid-point ke beech hua:<br/>(Trade Price > NBBO Bid ) & (Trade Price < NBBO Mid )                                                                                                                       |
| 43  | `TradeAtMid`            | Q,T  | Number                        | 0         | Us trade volume ka Sum jo mid par hua.<br/>TradePrice = NBBO MidPoint                                                                                                                                                                             |
| 44  | `TradeAtMidAsk`         | Q,T  | Number                        | 0         | Us ask volume ka Sum jo mid aur ask ke beech hua:<br/>(Trade Price > NBBO Mid) & (Trade Price < NBBO Ask)                                                                                                                                     |
| 45  | `TradeAtAsk`            | Q,T  | Number                        | 0         | Us trade volume ka Sum jo Ask par ya usse upar hua.                                                                                                                                                                                           |
| 46  | `TradeAtCrossOrLocked`  | Q,T  | Number                        | 0         | Bar ke liye trade volume ka Sum jab national best bid/offer locked ya crossed ho. <br>Locked matlab Bid = Ask <br>Crossed matlab Bid > Ask                                                                                                                  |
| 47  | `Volume`                | T    | Number                        | 0         | Trade hue shares ki total sankhya                                                                                                                                                                                                                    |
| 48  | `TotalTrades`           | T    | Number                        | 0         | Trades ki total sankhya                                                                                                                                                                                                                           |
| 49  | `FinraVolume`           | T    | Number                        | 0         | FINRA dwara report kiye gaye trade huye shares ki sankhya. <br/>FINRA dwara report kiye gaye trades broker-dealer internalization, dark pools, Over-The-Counter, aadi se hote hain.<br/>FINRA trades wo volume dikhate hain jo hidden hai ya trade ke liye publicly available nahi hai.         |
| 50  | `UptickVolume`          | T    | Integer                       | 0         | Bar ke dauran upticks ke saath trade huye shares ki total sankhya.<br/>Uptick = ( trade price > last trade price )                                                                                                                                                                                                                               |
| 51  | `DowntickVolume`        | T    | Integer                       | 0         | Bar ke dauran downticks ke saath trade huye shares ki total sankhya.<br/>Downtick = ( trade price < last trade price )                                                                                                                                                                                                                            |
| 52  | `RepeatUptickVolume`    | T    | Integer                       | 0         | Un shares ki total sankhya jahan trade price same (repeated) hai aur bar ke dauran pichla price change up tha. <br/>Repeat uptick = ( trade price == last trade price ) & (last tick direction == up )                                                                                                                                         |
| 53  | `RepeatDowntickVolume`  | T    | Integer                       | 0         | Un shares ki total sankhya jahan trade price same (repeated) hai aur bar ke dauran pichla price change down tha. <br/>Repeat downtick = ( trade price == last trade price ) & (last tick direction == down )                                                                                                                                   |
| 54  | `UnknownVolume`         | T    | Integer                       | 0         | Jab din ka pehla trade hota hai, to tick direction “unknown” hoti hai kyunki compare karne ke liye koi pichla Trade nahi hota.<br/>Ye field 4am ke baad pehle trade ka volume hai aur tick volume directions ke liye shuruaati value (initiation value) ke roop mein kaam karta hai.<br/>Future mein is bar ka naam badal kar `UnkownTickDirectionVolume` kar diya jayega.  |

### Notes

**Empty Fields**

Ek empty field ki koi value nahi hoti aur wo “Blank” hota hai, example ke liye FirstTradeTime aur bar period ke dauran koi trade nahi hui.
Bar mein trade huye total shares ko mapne wala field `Volume` `0` hoga agar koi Trades nahi huin (har field ke liye upar `No Value` column dekhein).

**No Bid/Ask/Trade OHLC**

Bar timeframe ke dauran ho sakta hai ki NBBO ya actual Trade mein koi badlav na ho.
Example ke liye, aisa bar ho sakta hai jisme OHLC Bid/Ask to ho par Trade OHLC na ho.

**Single Event**

Jin bars mein sirf ek trade, ek NBBO bid ya ek NBBO ask ho, unme Open/High/Low/Close price, size aur time same honge.

**`AskPriceAtHighBidPrice`, `AskSizeAtHighBidPrice`, `AskPriceAtLowBidPrice`, `AskSizeAtLowBidPrice` Fields** 

Kisi point in time par consistent Bid/Ask prices dikhane ke liye (jabki bar ke liye low/high Bid/Ask dikha rahe hon), AlgoSeek us price par low/high `Bid` aur corresponding `Ask` use karta hai.

## FAQ

**Trade Prices aksar Bid Price aur Ask Price range ke andar kyu hoti hain?**

Low/High Bid/Ask bar range ke liye low aur high NBBO price hai.
Aksar aisa hota hai ki Trade in prices par na ho kyunki price shayad sirf kuch seconds ke liye raha ho, ya executions mid-point par cross ho rahe hon hidden order types ki wajah se jo mid-point par execute hote hain ya current `Bid`/`Ask` par price improvement ki wajah se.

**Exchange tradable shares kaise milein?** 

Bar mein exchange tradable volume paane ke liye `FinraVolume` ko `Volume` se subtract karein.
- `Volume` trade huye shares ki total sankhya hai.
- ``FinraVolume`` un shares ki total sankhya hai jo FINRA dwara executions ke roop mein report kiye gaye hain.

Jab koi trade listed exchanges se hatkar hota hai, to use brokerage firm ya dark pool dwara FINRA ko report karna zaruri hota hai. Examples mein shamil hain:
- broker dealer dwara internal crosses
- over-the-counter block trades, aur
- dark pool executions.
