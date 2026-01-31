# Market Data ke saath kaam karna: NASDAQ_TotalView-ITCH Order Book

Halaanki FIX ka market share bahut bada hai, exchanges apne native protocols bhi offer karte hain. Nasdaq ek TotalView ITCH direct data-feed protocol deta hai jo subscribers ko equity instruments ke individual orders ko track karne ki suvidha deta hai - order lagne se lekar execute hone ya cancel hone tak.

Natijan, ye order book ko phir se banane (reconstruction) mein madad karta hai jo kisi specific security ya financial instrument ke active buy aur sell orders ki list ka hisab rakhta hai. Order book har price point par kitne shares ki boli (bid) ya offer lagayi gayi hai, ye dikhakar din bhar ki market depth batata hai. Ye specific buy aur sell orders ke piche kaunsa market participant hai, ye bhi pehchan sakta hai (agar order anonymously nahi lagaya gaya ho). Market depth liquidity ka aur bade market orders ka price par kya asar padega, iska ek main indicator hai.

Market aur limit orders ko match karne ke alawa, Nasdaq auctions ya crosses bhi operata karta hai jo market khulne aur band hone par badi sankhya mein trades execute karte hain. Crosses ka mahatva badhta ja raha hai kyunki passive investing badh rahi hai aur traders bade blocks of stock ko execute karne ke mauke dhundhte hain. TotalView Nasdaq opening aur closing crosses aur Nasdaq IPO/Halt cross ke liye Net Order Imbalance Indicator (NOII) bhi deta hai.

> Is example ke liye kaafi memory ki zarurat hai, shayad 16GB se zyada (main 64GB use kar raha hoon). Agar aapke paas itni capacity nahi hai, to dhyan rakhein ki is code ko chalana is book ki baaki cheezon ke liye zaruri nahi hai. Sabse pehle, iska maksad ye dikhana hai ki institutional investment context mein aap kis tarah ke data ke saath kaam karenge, jahan systems is single-day example se kahin bade data ko manage karne ke liye banaye gaye hote hain.

## Binary ITCH Messages ko Parse karna

ITCH v5.0 specification mein 20 se zyada message types hain jo system events, stock characteristics, limit orders lagane aur badalne, aur trade execution se jude hain. Isme open aur closing cross se pehle net order imbalance ki jaankari bhi hoti hai.

Nasdaq kai mahino ke daily binary files ke samples offer karta hai. Is chapter ke GitHub repository mein ek notebook, `build_order_book.ipynb` hai jo dikhata hai ki ITCH messages ki sample file ko kaise parse karein aur kisi bhi diye gaye tick ke liye executed trades aur order book dono ko kaise reconstruct karein.

Niche di gayi table book mein use ki gayi sample file (March 29, 2018) ke liye sabse aam message types ki frequency dikhati hai. Code ab March 27, 2019 ke naye sample ko use karne ke liye update kar diya gaya hai.

| Message type | Order book impact                                                                  | Number of messages |
|:------------:|------------------------------------------------------------------------------------|-------------------:|
| A            | New unattributed limit order                                                       | 136,522,761        |
| D            | Order canceled                                                                     | 133,811,007        |
| U            | Order canceled and replaced                                                        | 21,941,015         |
| E            | Full or partial execution; possibly multiple messages for the same original order  | 6,687,379          |
| X            | Modified after partial cancellation                                                | 5,088,959          |
| F            | Add attributed order                                                               | 2,718,602          |
| P            | Trade Message (non-cross)                                                          | 1,120,861          |
| C            | Executed in whole or in part at a price different from the initial display price   | 157,442            |
| Q            | Cross Trade Message                                                                | 17,233             |

Har message ke liye, specification components aur unki length aur data types ko batata hai:

| Name                    | Offset  | Length  | Value      | Notes                                                                                |
|-------------------------|---------|---------|------------|--------------------------------------------------------------------------------------|
| Message Type            | 0       | 1       | S          | System Event Message                                                                 |
| Stock Locate            | 1       | 2       | Integer    | Always 0                                                                             |
| Tracking Number         | 3       | 2       | Integer    | Nasdaq internal tracking number                                                      |
| Timestamp               | 5       | 6       | Integer    | Nanoseconds since midnight                                                           |
| Order Reference Number  | 11      | 8       | Integer    | The unique reference number assigned to the new order at the time of receipt.        |
| Buy/Sell Indicator      | 19      | 1       | Alpha      | The type of order being added. B = Buy Order. S = Sell Order.                        |
| Shares                  | 20      | 4       | Integer    | The total number of shares associated with the order being added to the book.        |
| Stock                   | 24      | 8       | Alpha      | Stock symbol, right padded with spaces                                               |
| Price                   | 32      | 4       | Price (4)  | The display price of the new order. Refer to Data Types for field processing notes.  |
| Attribution             | 36      | 4       | Alpha      | Nasdaq Market participant identifier associated with the entered order               |

Ye notebooks [01_build_itch_order_book](01_parse_itch_order_flow_messages.ipynb) (iska Hindi version bhi available hai), [02_rebuild_nasdaq_order_book](02_rebuild_nasdaq_order_book.ipynb) aur [03_normalize_tick_data](03_normalize_tick_data.ipynb) mein ye code shamil hai:
- NASDAQ Total View sample tick data download karne ke liye,
- binary source data se messages parse karne ke liye,
- kisi diye gaye stock ke liye order book reconstruct karne ke liye,
- order flow data ko visualize karne ke liye,
- tick data ko normalize karne ke liye.

Code ko March 27, 2019 ki latest NASDAQ sample file use karne ke liye update kiya gaya hai.

Warning: tick data ka size lagbhag 12GB hai aur kuch processing steps 4-core i7 CPU aur 32GB RAM par kai ghante le sakte hain.

## Tick data ko Regularize karna

Trade data nanoseconds ke hisab se indexed hota hai aur isme kaafi shor (noise) hota hai. Example ke liye, 'bid-ask bounce' ki wajah se price bid aur ask prices ke beech jhulati rehti hai jab trade ki shuruwat buy aur sell market orders ke beech badalti rehti hai. Noise-signal ratio ko sudharne aur statistical properties ko behtar banane ke liye, humein trading activity ko aggregate karke tick data ko 'resample' aur 'regularize' karne ki zarurat hoti hai.

Hum aam taur par aggregated period ke liye open (first), low, high, aur closing (last) price collect karte hain, saath hi volume-weighted average price (VWAP), trade huve shares ki sankhya, aur data se juda timestamp bhi.

Notebook [03_normalize_tick_data](03_normalize_tick_data.ipynb) dikhata hai ki alag-alag aggregation methods wale time aur volume bars ka use karke noisy tick data ko kaise normalize kiya jaye.
