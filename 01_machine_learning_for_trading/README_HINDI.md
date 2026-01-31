# Machine Learning for Trading: Idea se Execution tak (Ek Poori Guide)

Algorithmic trading un computer programs par nirbhar karta hai jo ek trading strategy ke kuch ya sabhi hisson ko automatic banate hain. **Algorithms** (Algorithm) ka matlab hai steps ya rules (niyam) ka ek sequence jo kisi goal ko paane ke liye banaya gaya hai. Ye kai tarah ke ho sakte hain aur investment process ke har hisse ko behtar banane mein madad karte hain - jaise trading updates sochne se lekar, paisa kahan lagana hai (asset allocation), trade lagana (execution), aur risk ko manage karna.

**Machine Learning (ML)** mein aise algorithms hote hain jo data se rules ya patterns (tarike) seekhte hain taaki wo kisi goal ko haasil kar sakein, jaise ki future prediction mein hone wali galti ko kam karna. Is book ke examples yeh dikhayenge ki kaise ML algorithms data se zaruri jaankari nikal sakte hain taaki wo investment ki mukhya गतिविधियों (activities) mein madad kar sakein ya unhe automatic kar sakein. In activities mein shamil hain:
*   Market ko dhyan se dekhna aur data ko analyze karke future ke baare mein andaza lagana.
*   Buy (khareedne) ya Sell (bechne) ke orders dene ka faisla lena.
*   Portfolio ko is tarah manage karna ki risk ke muqable achha return mile.

Aakhirkar, active investment management ka main goal **Alpha** generate karna hota hai. Alpha ka matlab hai wo return jo aapke benchmark (jise aap compare karne ke liye use karte hain) se zyada ho. **Fundamental Law of Active Management** (Active Management ka mul niyam) yeh kehta hai ki Alpha generate karne ki key (kunj) yeh hai ki aapke paas:
1.  Return ke sahi forecasts (bhavishyavani) hon.
2.  In forecasts par action lene ki ability ho. (Grinold 1989; Grinold and Kahn 2000).

Yeh **Information Ratio (IR)** ko define karta hai, jo active management ki value batata hai. IR nikalne ke liye portfolio aur benchmark ke return ke antar (difference) ko un returns ki volatility (utar-chadhaav) se divide kiya jata hai. Iska formula kuch aisa samjha ja sakta hai:
*   **Information Coefficient (IC):** Yeh mapta hai ki aapke forecast (andaze) kitne sahi hain (rank correlation ke through).
*   **Breadth of a Strategy (Strategy ki chaudai):** Iska square root, jo dikhata hai ki aapne in forecasts par kitne alag-alag daav (independent bets) lagaye hain.

Financial market mein bahut samajhdar investors ke beech competition ka matlab hai ki Alpha generate karne ke liye aapko "superior information" (behtar jaankari) chahiye. Ya to aapke paas behtar data ho, ya us data ko process karne ki behtar kabiliyat, ya phir dono. Yahi par **ML** kaam aata hai: **ML for Trading (ML4T)** ka maksad yahi hai ki tezi se badhte hue data ka behtar use kiya jaye taaki aise forecast milen jo accurate bhi hon aur jin par action bhi liya ja sake. Isse investment decisions aur results ki quality behtar hoti hai.

Pehle, algorithmic trading ka matlab sirf trade execution ko automate karna hota tha taaki kharcha kam ho sake. Lekin yeh book ek bada nazariya leti hai kyunki ab algorithms aur ML ka use bahut saari cheezon mein hone laga hai - jaise naye ideas dhundhna, data se signals nikalna, ye tay karna ki kitna paisa kahan lagana hai (position-sizing), aur strategies ko test aur evaluate karna.

Yeh chapter un industry trends ko dekhta hai jinki wajah se ML investment industry mein ek competitive advantage ban gaya hai. Hum yeh bhi dekhenge ki ek trading strategy banane ke poore process mein ML kahan fit hota hai.

## Content (Vishay Soochi)

1. [Investment industry mein ML ka uday](#the-rise-of-ml-in-the-investment-industry)
    * [Electronic se High-Frequency Trading tak](#from-electronic-to-high-frequency-trading)
    * [Factor investing aur Smart Beta funds](#factor-investing-and-smart-beta-funds)
    * [Algorithmic pioneers ne humans ko peeche chhoda](#algorithmic-pioneers-outperform-humans)
        - [ML driven funds ne $1 trillion AUM attract kiya](#ml-driven-funds-attract-1-trillion-aum)
        - [Quantamental funds ka uday](#the-emergence-of-quantamental-funds)
    * [ML aur Alternative Data](#ml-and-alternative-data)
2. [Ek ML-driven strategy ko design aur execute karna](#designing-and-executing-an-ml-driven-strategy)
    * [Data lana aur manage karna (Sourcing and managing data)](#sourcing-and-managing-data)
    * [Alpha factor research se portfolio management tak](#from-alpha-factor-research-to-portfolio-management)
    * [Strategy Backtesting](#strategy-backtesting)
3. [Practice mein ML for Trading: Strategies aur Use Cases](#ml-for-trading-in-practice-strategies-and-use-cases)
    * [Algorithmic strategies ka vikas (Evolution)](#the-evolution-of-algorithmic-strategies)
    * [Trading ke liye ML ke use cases](#use-cases-of-ml-for-trading)
        - [Insights aur features nikalne ke liye Data Mining](#data-mining-for-feature-extraction-and-insights)
        - [Alpha factor banane aur milane ke liye Supervised Learning](#supervised-learning-for-alpha-factor-creation-and-aggregation)
        - [Asset Allocation](#asset-allocation)
        - [Trade ideas ko test karna](#testing-trade-ideas)
        - [Reinforcement Learning](#reinforcement-learning)
4. [Resources & References (Sansadhan aur Sandarbh)](#resources--references)
    * [Academic Research](#academic-research)
    * [Industry News](#industry-news)
    * [Books](#books)
        - [Machine Learning](#machine-learning)
    * [Courses](#courses)
    * [ML Competitions & Trading](#ml-competitions--trading)
    * [Python Libraries](#python-libraries)

## The rise of ML in the investment industry (Investment Industry mein ML ka uday)

Pichle kuch dashakon (decades) mein investment industry dramayi roop se badli hai aur yeh badlav ab bhi jari hai. Competition badh raha hai, technology aage badh rahi hai, aur economic mahaul chunautipurn hai. Yeh section un mukhya trends ko review karta hai jinhone investment environment ko shape diya hai, aur khaaskar algorithmic trading aur ML ke use ko badhaya hai.

Wo trends jinhone algorithmic trading aur ML ko aaj itna popular bana diya hai, wo hain:
- **Market Microstructure mein badlav:** Jaise electronic trading ka failna aur alag-alag asset classes aur deshon ke markets ka aapas mein judna.
- **Investment Strategies ka vikas:** Ab log sirf asset classes (jaise Stocks vs Bonds) ke bajaye 'Risk Factor Exposure' ke hisab se sochte hain.
- **Computing Power aur Data ki kranti:** Computers tej ho gaye hain, data bahut sara hai, aur statistical methods (jaise Deep Learning) mein nayi discoveries hui hain.
- **Algorithms ki jeet:** Algorithmic trading shuru karne wale pioneers ne insani investors (discretionary investors) se behtar perform kiya hai.

Iske alawa, 2001 aur 2008 ke financial crisis ne investors ko yeh sochne par majboor kiya ki wo risk ko kaise manage karein. Iska ek natija yeh hua ki log low-cost passive investment (jaise ETFs) ki taraf bhage. 2008 ke baad jab returns kam the aur market shant tha, tab cost bachane ke liye investors ne $3.5 trillion se zyada paisa Active Mutual Funds se nikal kar Passive ETFs mein daal diya.

Competition ka asar fees par bhi dikha. Hedge funds ki fees jo pehle "2% management fee aur 20% profit share" hoti thi, wo 2017 mein girkar average 1.48% aur 17.4% reh gayi.

### From electronic to high-frequency trading (Electronic se High-Frequency Trading tak)

1960s mein jab networks ne prices ko computer terminals tak bhejna shuru kiya tha, tabse lekar ab tak Electronic Trading bahut aage badh chuki hai - speed, volume, aur alag-alag markets ko cover karne mein.

*   [Dark Pool Trading & Finance](https://www.cfainstitute.org/en/advocacy/issues/dark-pools), CFA Institute
*   [Dark Pools in Equity Trading: Policy Concerns and Recent Developments](https://crsreports.congress.gov/product/pdf/R/R43739), Congressional Research Service, 2014
*   [High Frequency Trading: Overview of Recent Developments](https://fas.org/sgp/crs/misc/R44443.pdf), Congressional Research Service, 2016

### Factor investing and smart beta funds

Kisi asset se milne wala return us investment ke sath jude 'Risk' (johkim) ka phal hota hai. Example ke liye, agar aap kisi company ke share (equity) mein paisa lagate hain, to aap us company ke business fail hone ka risk le rahe hain. Agar bond khareedte hain, to default ka risk le rahe hain.

Agar kuch khaas tarah ke risk lene se return milta hai, to in 'Risk Factors' ko pehchanna aur unka forecast karna strategy design karne ka main focus ban jata hai. Yahi se humein valuable trading signals milte hain. Industry ki risk factors ki samajh waqt ke sath bahut gehrai gayi hai aur isne algorithmic trading mein ML ke use ko prabhavit kiya hai.

Jo factors CAPM (ek purana model) se alag returns ko explain karte the, unhe investment styles mein shamil kiya gaya. Log aise portfolios banane lage jo kisi ek factor (jaise 'Value' ya 'Momentum') ki taraf jhuke hon. 2008 ke crisis ne dikhaya ki sirf 'Asset Class' (jaise Gold, Stocks) ke naam par bharosa karna galat ho sakta hai, kyunki jab market girta hai to sabhi assets ek sath gir sakte hain. Asli cheez 'Factor Risk' hoti hai.

Pichle kuch dashakon mein, 'Quantitative Factor Investing' simple styles se badhkar complex 'Smart Beta' products tak pahunch gaya hai. Smart Beta funds ne 2017 mein $1 trillion assets cross kar liye. Yeh funds passive hote hain (yani baar-baar share khareed-bech nahi karte) lekin ye kisi factor ke hisab se modify kiye jate hain - jaise 'Saste Stocks' (Value) ya 'High Dividend' wale stocks chunna. Yeh growth isliye bhi hui kyunki log traditional managers ki high fees se pareshan the.

Risk factors ki khoj aur unka sahi forecast karna - jo akele ya dusre factors ke sath milkar future returns par asar dalte hain - yehi wajah hai ki investment mein ML ka use badh raha hai. Yeh is book ka ek main theme rahega.

### Algorithmic pioneers outperform humans (Algorithmic pioneers ne humans ko peeche chhoda)

Jin firms ne algorithmic trading ki shuruwat ki, unka track record aur unke paas manage karne ke liye paisa (AUM) itna badha ki poori industry unki nakal karne ki koshish karne lagi.

Systematic strategies (jo poori tarah algorithms par chalti hain) ko sabse pehle mathematician **James Simons** ne famous kiya. Unhone 1982 mein **Renaissance Technologies** banayi. Unka **Medallion Fund**, jo bahari logo ke liye band hai, ne 1982 se lekar ab tak lagbhag 35% saalana return diya hai.

DE Shaw, Citadel, aur Two Sigma - ye teen bade quantitative hedge funds hain jo algorithms use karte hain. 2017 mein ye pehli baar 'All-time top-20 performers' ki list mein aaye (investors ke liye kamaye gaye total dollars ke hisab se).

#### ML driven funds attract $1 trillion AUM

Morgan Stanley ne 2017 mein estimate kiya ki algorithmic strategies pichle 6 saalon mein 15% har saal ki dar se badhi hain aur karib $1.5 trillion control karti hain. Kuch reports ke mutabik Quantitative Hedge Fund industry $1 trillion AUM cross karne wali thi. Iske mukable, total hedge fund industry ki capital $3.21 trillion thi.

*   [Global Algorithmic Trading Market to Surpass US$ 21,685.53 Million by 2026](https://www.bloomberg.com/press-releases/2019-02-05/global-algorithmic-trading-market-to-surpass-us-21-685-53-million-by-2026)
*   [The stockmarket is now run by computers, algorithms and passive managers](https://www.economist.com/briefing/2019/10/05/the-stockmarket-is-now-run-by-computers-algorithms-and-passive-managers), Economist, Oct 5, 2019

#### The emergence of quantamental funds (Quantamental funds ka uday)

Active investment management mein do alag tarike (approaches) hain:
1.  **Systematic (Quant):** Jo algorithms aur data par nirbhar karte hain aur dher saare securities mein opportunities dhundhte hain.
2.  **Discretionary:** Jahan expert log kuch chuni hui securities ka gehra analysis karte hain.

Ab ye dono tarike mil rahe hain. Ise **"Quantamental"** (Quant + Fundamental) kehte hain. Fundamental managers bhi ab data science ki madad le rahe hain. Barclays ke mutabik, ab fundamental traders bhi quantitative techniques use kar rahe hain. Quants ab total hedge fund assets ka lagbhag 17% hissa hain.

### ML and alternative data (ML aur Vaikalpik Data)

Hedge funds hamesha 'Information Advantage' (kuch aisi jaankari jo dusron ke paas na ho) dhundhte rehte hain taaki Alpha mil sake. Pehle iska matlab hota tha shoppers ka survey karna, ya vote se pehle logon se poochna. Kabhi-kabhi log galat tareeke (insider trading) bhi use karte the, jiske liye 2010 ke baad kafi sakhti hui aur kai logon ko saza mili.

Iske ulat, ML aur alternative data ka use karke jo advantage milta hai, wo kisi insider information se nahi, balki **badi matra mein data collect karne aur use real-time mein analyze karne ki shamta** se aata hai.

Teen trends ne data ke use ko krantikari bana diya hai:
1.  Digital data ki matra mein bhari izafa (Exponential increase).
2.  Computing power aur storage ka sasta aur behtar hona.
3.  Complex datasets ko analyze karne ke liye ML methods mein tarakki.

*   [Can We Predict the Financial Markets Based on Google's Search Queries?](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.2446), Perlin, et al, 2016, Journal of Forecasting

## Designing and executing an ML-driven strategy (Ek ML-driven strategy ko design aur execute karna)

ML trading strategy ke lifecycle mein kai steps par value add kar sakta hai. Iske liye infrastructure aur data ki zarurat hoti hai. Yeh book yahi batati hai ki strategy ko design, execute aur evaluate karne ke process mein ML kaise fit hota hai.

Ek algorithmic trading strategy **Alpha Factors** ke combination se chalti hai. Alpha factors data sources ko signals mein badalte hain, jo future returns predict karte hain aur buy/sell orders trigger karte hain. Chapter 2 aur Chapter 3 mein hum **Data Sourcing aur Management** ke baare mein padhenge, jo kisi bhi strategy ka sabse zaruri 'Raw Material' hai.

[Chapter 4, Alpha Factor Research](../04_alpha_factor_research) ek sahi tarika batata hai false discoveries (galat khoj) se bachne ka. [Chapter 5, Strategy Evaluation](../05_strategy_evaluation) strategy ke execution aur performance ko mapne ka context deta hai.

Niche in steps ki outline di gayi hai, jinhe hum puri book mein detail mein samjhenge.

### Sourcing and managing data (Data lana aur manage karna)

Data ab bahut zyada, bahut tarah ka aur bahut tezi se mil raha hai. Yeh ML ke liye bahut achhi baat hai. Lekin is badhte hue data ko sambhalna aur sahi data chunna bahut zaruri hai. Isme ye steps shamil hain:

1.  Market, Fundamental, aur Alternative data sources ko pehchanna jisme Alpha signals hon jo jaldi kharab na hon.
2.  Cloud-based data infrastructure (jaise Hadoop ya Spark) use karna taaki data tezi se aur aasani se mile.
3.  Data ko **Point-in-Time** basis par manage karna taaki **Look-ahead bias** se bacha ja sake. Iska matlab hai ki data waisa hi hona chahiye jaisa us waqt asliyat mein tha. Agar aapne future ka data galti se training mein use kar liya (jo us waqt available nahi tha), to live trading mein aapki strategy pakka fail hogi.

Hum in cheezon ko Chapter 2 aur Chapter 3 mein detail mein cover karenge.

### From alpha factor research to portfolio management (Alpha factor research se portfolio management tak)

Alpha factors ko data se signal nikalne ke liye design kiya jata hai taaki wo asset returns predict kar sakein. Ek factor har asset ke liye ek value deta hai.

Trading strategy workflow ke **Research Phase** mein alpha factors ka design, evaluation, aur combination shamil hota hai. ML isme bada role play karta hai kyunki aajkal data bahut complex hai aur simple factors jaldi kaam karna band kar dete hain (signal decay).

Alpha factors se Entry aur Exit signals milte hain, jisse Buy ya Sell orders bante hain. In orders se apka Portfolio banta hai. **Portfolio Management** ka kaam hai position weights (kis share mein kitna paisa) ko optimize karna taaki jo risk aur return hum chahte hain wo mile. Yeh process dynamic hota hai aur naye market data ke hisab se badalta rehta hai.

### Strategy backtesting

Kisi bhi idea ko strategy mein badalne se pehle uska **Backtesting** karna zaruri hai. Yeh ek scientific tarika hai jisme hum purane data (ya simulated data) par strategy ko chala kar dekhte hain ki kya wo kaam karti hai ya fail ho jati hai. Hum koshish karte hain ki strategy ko reject karein (taki sirf best wali hi bachein). Simulated data un scenarios ko test karne mein madad karta hai jo ho sakte the par history mein nahi hue.

## ML for trading in practice: strategies and use cases (Practice mein ML for Trading)

Asli duniya mein, hum ML ko kisi specific business goal ke liye use karte hain. Is section mein hum dekhenge ki trading strategies kaise badli hain aur ML ke real-world examples kya hain.

### The evolution of algorithmic strategies (Algorithmic strategies ka vikas)

Quantitative strategies teen lehron (waves) mein vikasit hui hain:

1.  **1980s aur 1990s:** Signals aksar academic research se aate the aur inme ek ya do simple inputs hote the. AQR (ek bada hedge fund) 1998 mein bana tha jo aisi strategies use karta tha. Aajkal ye signals aam ho gaye hain aur ETFs ke roop mein milte hain.
2.  **2000s:** Factor-based investing badha (Fama aur French ke kaam ke baad). Funds algorithms use karke aise assets dhundhte the jo Value ya Momentum jaise risk factors se jude hon. August 2007 mein "Quant Quake" aaya jab bahut saare funds ek sath gir gaye. Aaj ye strategies 'Smart Beta' funds ke roop mein milti hain.
3.  **The Third Era (Ab):** Yeh ML capabilities aur Alternative Data par chalta hai. Yahan "Factor Decay" ek badi chunauti hai: naye ideas se milne wala profit, jaise hi wo public hote hain, tezi se kam ho jata hai kyunki sab log use karne lagte hain.

Aajkal traders algorithms ka use alag-alag maksad ke liye karte hain:
*   Trade execution (sahi daam par khareedna/bechna).
*   Short-term trades (chote price movements se profit kamana, arbitrage).
*   Behavioral strategies (dusre traders ke behavior ka andaza lagana).
*   Price aur return predictions par based strategies.

### Use cases of ML for trading

ML ka use algorithmic trading ke har step par ho sakta hai. Mukhya applications ye hain:
*   **Data Mining:** Patterns pehchanna, features nikalna aur insights lena.
*   **Supervised Learning:** Risk factors ya alpha generate karna aur trade ideas banana.
*   **Aggregation:** Alag-alag signals ko milakar ek strategy banana.
*   **Asset Allocation:** Algorithm ke dwara seekhe gaye risk profiles ke hisab se paisa batna.
*   **Testing:** Strategies ko evaluate karna (Synthetic data use karke).
*   **Reinforcement Learning:** Strategy ko automatically improve karna.

Hum inme se kuch applications ko niche highlight kar rahe hain aur batayenge ki ye book ke kaunse chapter mein milenge.

#### Data mining for feature extraction and insights

Bade aur complex datasets ko saste mein evaluate karne ke liye signals ko scale par detect karna zaruri hai.
*   **Information Theory:** Yeh batata hai ki kisi feature mein kitna 'Signal' hai. Chapter 4 mein hum 'Mutual Information' ka use karenge ye dekhne ke liye ki kaunsa feature asset returns predict karne ke liye sabse accha hai.
*   **Unsupervised Learning:** Bina kisi label ke data mein structure dhundhna.
    *   Chapter 13 mein hum 'Clustering' use karke high-dimensional data se features banayenge.
    *   Chapter 15 mein hum Financial Text data (news, earnings calls) ko summarize karne ke liye models use karenge.
    *   Chapter 20 mein hum Deep Learning (Autoencoders) use karke non-linear risk factors nikalenge.
*   **Model Transparency:** Hum 'SHAP' values ka use karenge ye samajhne ke liye ki model ko kaunsa variable sabse zyada important lag raha hai (Chapter 12).

#### Supervised learning for alpha factor creation and aggregation

ML ka sabse aam use hai predictions, price movements, ya fundamentals ka anuman lagana.
*   **Downstream Models:** Alag-alag predictions ko milakar portfolio level par signal dete hain.
*   ML predictions **Discretionary Trades** (insani faislon) ko bhi support kar sakti hain (Quantamental approach).

ML predictions specific risk factors (jaise Volatility) ko target kar sakti hain:
*   Chapter 3 mein hum fundamental data se valuation models banayenge.
*   Chapter 14, 15, aur 16 mein hum text data (news, reviews) se company ka revenue predict karenge.
*   Chapter 9 mein hum Time Series models use karke volatility forecast karenge.
*   Chapter 19 mein hum RNNs (Recurrent Neural Networks) use karenge jo time-series data ke sath accha kaam karte hain.

#### Asset allocation
ML ka use portfolios ko allocate karne ke liye kiya gaya hai, jahan risk characteristics "Asset Classes" ke bajaye "Asset Prices ke patterns" se tay hoti hain.
*   Chapter 5 aur 13 mein hum dikhayenge ki kaise 'Hierarchical Clustering' traditional tarikon se behtar risk classes banata hai.

#### Testing trade ideas
Backtesting strategies chunne ke liye critical hai. Cross-validation aur synthetic data ka use karke hum bharosemand results pa sakte hain.
*   Financial data time-series hota hai (samay ke sath chalta hai), isliye standard testing methods ko change karna padta hai taaki hum future ka data galti se use na karein.
*   Chapter 21 mein hum **GANs (Generative Adversarial Networks)** ka use karke nakli (synthetic) training data banayenge taaki humare models ko aur data mile seekhne ke liye.

#### Reinforcement learning
Trading ek competitive game jaisa hai. Reinforcement learning mein hum agents (bots) ko train karte hain ki wo rewards (munafa) ke hisab se khud seekhein. Yeh financial ML ka sabse promising area mana jata hai.
*   Chapter 22 mein hum **Deep Reinforcement Learning** aur Q-learning algorithms use karke ek trading agent banayenge OpenAI Gym environment mein.

## Resources & References (Sansadhan aur Sandarbh)

### Academic Research
*   [The fundamental law of active management](http://jpm.iijournals.com/content/15/3/30), Richard C. Grinold, The Journal of Portfolio Management Spring 1989, 15 (3) 30-37
*   [The relationship between return and market value of common stocks](https://www.sciencedirect.com/science/article/pii/0304405X81900180), Rolf Banz,Journal of Financial Economics, March 1981
*   [The Arbitrage Pricing Theory: Some Empirical Results](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1981.tb00444.x), Marc Reinganum, Journal of Finance, 1981
*   [The Relationship between Earnings' Yield, Market Value and Return for NYSE Common Stock](https://pdfs.semanticscholar.org/26ab/311756099c8f8c4e528083c9b90ff154f98e.pdf), Sanjoy Basu, Journal of Financial Economics, 1982
*   [Bridging the divide in financial market forecasting: machine learners vs. financial economists](http://www.sciencedirect.com/science/article/pii/S0957417416302585), Expert Systems with Applications, 2016
*   [Financial Time Series Forecasting with Deep Learning : A Systematic Literature Review: 2005-2019](http://arxiv.org/abs/1911.13288), arXiv:1911.13288 [cs, q-fin, stat], 2019
*   [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009), The Review of Financial Studies, 2020
*   [The Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns](http://academic.oup.com/rfs/article/30/12/4389/3091648), The Review of Financial Studies, 2017
*   [Characteristics are covariances: A unified model of risk and return](http://www.sciencedirect.com/science/article/pii/S0304405X19301151), Journal of Financial Economics, 2019
*   [Estimation and Inference of Heterogeneous Treatment Effects using Random Forests](https://doi.org/10.1080/01621459.2017.1319839), Journal of the American Statistical Association, 2018
*   [An Empirical Study of Machine Learning Algorithms for Stock Daily Trading Strategy](https://www.hindawi.com/journals/mpe/2019/7816154/), Mathematical Problems in Engineering, 2019
*   [Predicting stock market index using fusion of machine learning techniques](http://www.sciencedirect.com/science/article/pii/S0957417414006551), Expert Systems with Applications, 2015
*   [Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques](http://www.sciencedirect.com/science/article/pii/S0957417414004473), Expert Systems with Applications, 2015
*   [Deep Learning for Limit Order Books](http://arxiv.org/abs/1601.01987), arXiv:1601.01987 [q-fin], 2016
*   [Trading via Image Classification](http://arxiv.org/abs/1907.10046), arXiv:1907.10046 [cs, q-fin], 2019
*   [Algorithmic trading review](http://doi.org/10.1145/2500117), Communications of the ACM, 2013
*   [Assessing the impact of algorithmic trading on markets: A simulation approach](https://www.econstor.eu/handle/10419/43250), , 2008
*   [The Efficient Market Hypothesis and Its Critics](http://www.aeaweb.org/articles?id=10.1257/089533003321164958), Journal of Economic Perspectives, 2003
*   [The Arbitrage Pricing Theory Approach to Strategic Portfolio Planning](https://doi.org/10.2469/faj.v40.n3.14), Financial Analysts Journal, 1984

### Industry News (Industry Samachar)

*   [The Rise of the Artificially Intelligent Hedge Fund](https://www.wired.com/2016/01/the-rise-of-the-artificially-intelligent-hedge-fund/#comments), Wired, 25-01-2016
*   [Crowd-Sourced Quant Network Allocates Most Ever to Single Algo](https://www.bloomberg.com/news/articles/2018-08-02/crowd-sourced-quant-network-allocates-most-ever-to-single-algo), Bloomberg, 08-02-2018
*   [Goldman Sachs’ lessons from the ‘quant quake’](https://www.ft.com/content/fdfd5e78-0283-11e7-aa5b-6bb07f5c8e12), Financial Times, 03-08-2017
*   [Lessons from the Quant Quake resonate a decade later](https://www.ft.com/content/a7a04d4c-83ed-11e7-94e2-c5b903247afd), Financial Times, 08-18-2017
*   [Smart beta funds pass $1tn in assets](https://www.ft.com/content/bb0d1830-e56b-11e7-8b99-0191e45377ec), Financial Times, 12-27-2017
*   [BlackRock bets on algorithms to beat the fund managers](https://www.ft.com/content/e689a67e-2911-11e8-b27e-cc62a39d57a0), Financial Times, 03-20-2018
*   [Smart beta: what’s in a name?](https://www.ft.com/content/d1bdabaa-a9f0-11e7-ab66-21cc87a2edde), Financial Times, 11-27-2017
*   [Computer-driven hedge funds join industry top performers](https://www.ft.com/content/9981c870-e79a-11e6-967b-c88452263daf), Financial Times, 02-01-2017
*   [Quants Rule Alpha’s Hedge Fund 100 List](https://www.institutionalinvestor.com/article/b1505pmf2v2hg3/quants-rule-alphas-hedge-fund-100-list), Institutional Investor, 06-26-2017
*   [The Quants Run Wall Street Now](https://www.wsj.com/articles/the-quants-run-wall-street-now-1495389108), Wall Street Journal, 05-21-2017
*   ['We Don’t Hire MBAs': The New Hedge Fund Winners Will Crunch The Better Data Sets](https://www.cbinsights.com/research/algorithmic-hedge-fund-trading-winners/), cbinsights, 06-28-2018
*   [Artificial Intelligence: Fusing Technology and Human Judgment?](https://blogs.cfainstitute.org/investor/2017/09/25/artificial-intelligence-fusing-technology-and-human-judgment/), CFA Institute, 09-25-2017
*   [The Hot New Hedge Fund Flavor Is 'Quantamental'](https://www.bloomberg.com/news/articles/2017-08-25/the-hot-new-hedge-fund-flavor-is-quantamental-quicktake-q-a), Bloomberg, 08-25-2017
*   [Robots Are Eating Money Managers’ Lunch](https://www.bloomberg.com/news/articles/2017-06-20/robots-are-eating-money-managers-lunch), Bloomberg, 06-20-2017
*   [Rise of Robots: Inside the World's Fastest Growing Hedge Funds](https://www.bloomberg.com/news/articles/2017-06-20/rise-of-robots-inside-the-world-s-fastest-growing-hedge-funds), Bloomberg, 06-20-2017
*   [When Silicon Valley came to Wall Street](https://www.ft.com/content/ba5dc7ca-b3ef-11e7-aa26-bb002965bce8), Financial Times, 10-28-2017
*   [BlackRock bulks up research into artificial intelligence](https://www.ft.com/content/4f5720ce-1552-11e8-9376-4a6390addb44), Financial Times, 02-19-2018
*   [AQR to explore use of ‘big data’ despite past doubts](https://www.ft.com/content/3a8f69f2-df34-11e7-a8a4-0a1e63a52f9c), Financial Times, 12-12-2017
*   [Two Sigma rapidly rises to top of quant hedge fund world](https://www.ft.com/content/dcf8077c-b823-11e7-9bfb-4a9c83ffa852), Financial Times, 10-24-2017
*   [When Silicon Valley came to Wall Street](https://www.ft.com/content/ba5dc7ca-b3ef-11e7-aa26-bb002965bce8), Financial Times, 10-28-2017
*   [Artificial intelligence (AI) in finance - six warnings from a central banker](https://www.bundesbank.de/en/press/speeches/artificial-intelligence--ai--in-finance--six-warnings-from-a-central-banker-711602), Deutsche Bundesbank, 02-27-2018
*   [Fintech: Search for a super-algo](https://www.ft.com/content/5eb91614-bee5-11e5-846f-79b0e3d20eaf), Financial Times, 01-20-2016
*   [Barron’s Top 100 Hedge Funds](https://www.barrons.com/articles/top-100-hedge-funds-1524873705)
*   [How high-frequency trading hit a speed bump](https://www.ft.com/content/d81f96ea-d43c-11e7-a303-9060cb1e5f44), FT, 01-01-2018

### Books (Kitabein)

*   [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086), Marcos Lopez de Prado, 2018
*   [Quantresearch](http://www.quantresearch.info/index.html) by Marcos López de Prado
*   [Quantitative Trading](http://epchan.blogspot.com/), Ernest Chan
*   [Machine Learning in Finance](https://www.springer.com/gp/book/9783030410674), Dixon, Matthew F., Halperin, Igor, Bilokon, Paul, Springer, 2020

#### Machine Learning

*   [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html), Tom Mitchell, McGraw Hill, 1997
*   [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), Gareth James et al.
    *   Essential machine learning concepts ke liye behtareen reference, online free available hai.
*   [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf), Barber, D., Cambridge University Press, 2012 (updated version author ki website par available hai)

### Courses (Courses)

*   [Algorithmic Trading](http://personal.stevens.edu/~syang14/fe670.htm), Prof. Steve Yang, Stevens Institute of Technology
*   [Machine Learning](https://www.coursera.org/learn/machine-learning), Andrew Ng, Coursera
*   [Deep Learning Specialization](http://deeplearning.ai/), Andrew Ng
    *   Andrew Ng ka introductory deep learning course
*   Machine Learning for Trading Specialization, [Coursera](https://www.coursera.org/specializations/machine-learning-trading)
*   Machine Learning for Trading, Georgia Tech CS 7646, [Udacity](https://www.udacity.com/course/machine-learning-for-trading--ud501)
*   Introduction to Machine Learning for Trading, [Quantinsti](https://quantra.quantinsti.com/course/introduction-to-machine-learning-for-trading)

### ML Competitions & Trading (ML Pratiyogitayein aur Trading)

*   [IEEE Investment Ranking Challenge](https://www.crowdai.org/challenges/ieee-investment-ranking-challenge)
    *   [Investment Ranking Challenge : Semi-annual returns ke basis par best performing stocks ko pehchanna](https://arxiv.org/pdf/1906.08636.pdf)
*   [Two Sigma Financial Modeling Challenge](https://www.kaggle.com/c/two-sigma-financial-modeling)
*   [Two Sigma: Using News to Predict Stock Movements](https://www.kaggle.com/c/two-sigma-financial-news)
*   [The Winton Stock Market Challenge](https://www.kaggle.com/c/the-winton-stock-market-challenge)
*   [Algorithmic Trading Challenge](https://www.kaggle.com/c/AlgorithmicTradingChallenge)

### Python Libraries

*   matplotlib [docs](https://github.com/matplotlib/matplotlib)
*   numpy [docs](https://github.com/numpy/numpy)
*   pandas [docs](https://github.com/pydata/pandas)
*   scipy [docs](https://github.com/scipy/scipy)
*   scikit-learn [docs](https://scikit-learn.org/stable/user_guide.html)
*   LightGBM [docs](https://lightgbm.readthedocs.io/en/latest/)
*   CatBoost [docs](https://catboost.ai/docs/concepts/about.html)
*   TensorFlow [docs](https://www.tensorflow.org/guide)
*   PyTorch [docs](https://pytorch.org/docs/stable/index.html)
*   Machine Learning Financial Laboratory (mlfinlab) [docs](https://mlfinlab.readthedocs.io/en/latest/)
*   seaborn [docs](https://github.com/mwaskom/seaborn)
*   statsmodels [docs](https://github.com/statsmodels/statsmodels)
*   [Boosting numpy: Why BLAS Matters](http://markus-beuckelmann.de/blog/boosting-numpy-blas.html)
