# Machine Learning ke liye Trading: From Idea to Execution

Algorithmic trading relies on computer programs that execute algorithms to automate some or all elements ka a trading strategy. **Algorithms** hain a sequence ka steps or rules designed to achieve a goal. They can take many forms aur facilitate optimization throughout the investment process, from idea generation to asset allocation, trade execution, aur risk management.

**Machine learning** (ML) involves algorithms that learn rules or patterns from data to achieve a goal such as minimizing a prediction error. The examples mein this book will illustrate how ML algorithms can extract information from data to support or automate key investment activities. These activities include observing the market aur analyzing data to form expectations about the future aur decide on placing buy or sell orders, as well as managing the resulting portfolio to produce attractive returns relative to the risk.

Ultimately, the goal ka active investment management hai to generate alpha, defined as portfolio returns mein excess ka the benchmark used ke liye evaluation. The **fundamental law ka active management** postulates that the key to generating alpha hai having accurate return forecasts combined ke saath the ability to act on these forecasts (Grinold 1989; Grinold aur Kahn 2000).

It defines the **information ratio** (IR) to express the value ka active management as the ratio ka the return difference between the portfolio aur a benchmark to the volatility ka those returns. It further approximates the IR as the product ka
- The **information coefficient** (IC), which measures the quality ka forecast as their rank correlation ke saath the outcomes
- The square root ka the **breadth ka a strategy** expressed as the number ka independent bets on these forecasts

The competition ka sophisticated investors mein financial markets implies that making precise predictions to generate alpha requires superior information, either through access to better data, a superior ability to process it, or both. Yeh hai where ML comes mein: applications ka **ML ke liye trading (ML4T)** typically aim to make more efficient use ka a rapidly diversifying range ka data to produce both better aur more actionable forecasts, thus improving the quality ka investment decisions aur results.

Historically, algorithmic trading used to be more narrowly defined as the automation ka trade execution to minimize the costs offered by the sell-side. Yeh book takes a more comprehensive perspective since the use ka algorithms mein general aur ML, mein particular, has come to impact a broader range ka activities from generating ideas aur extracting signals from data to asset allocation, position-sizing, aur testing aur evaluating strategies.

Yeh chapter looks at industry trends that have led to the emergence ka ML as a source ka competitive advantage mein the investment industry. hum will also look at where ML fits into the investment process to enable algorithmic trading strategies. 

## Vishay-suchi (Content)

1. [The rise ka ML mein the investment industry](#the-rise-ka-ml-mein-the-investment-industry)
    * [From electronic to high-frequency trading](#from-electronic-to-high-frequency-trading)
    * [Factor investing and smart beta funds](#factor-investing-and-smart-beta-funds)
    * [Algorithmic pioneers outperform humans](#algorithmic-pioneers-outperform-humans)
        - [ML driven funds attract $1 trillion AUM](#ml-driven-funds-attract-1-trillion-aum)
        - [The emergence of quantamental funds](#the-emergence-of-quantamental-funds)
    * [ML and alternative data](#ml-and-alternative-data)
2. [Designing aur executing an ML-driven strategy](#designing-aur-executing-an-ml-driven-strategy)
    * [Sourcing and managing data](#sourcing-and-managing-data)
    * [From alpha factor research to portfolio management](#from-alpha-factor-research-to-portfolio-management)
    * [Strategy backtesting](#strategy-backtesting)
3. [ML ke liye trading mein practice: strategies aur use cases](#ml-ke liye-trading-mein-practice-strategies-aur-use-cases)
    * [The evolution of algorithmic strategies](#the-evolution-of-algorithmic-strategies)
    * [Use cases of ML for trading](#use-cases-of-ml-for-trading)
        - [Data mining for feature extraction and insights](#data-mining-for-feature-extraction-and-insights)
        - [Supervised learning for alpha factor creation and aggregation](#supervised-learning-for-alpha-factor-creation-and-aggregation)
        - [Asset allocation](#asset-allocation)
        - [Testing trade ideas](#testing-trade-ideas)
        - [Reinforcement learning](#reinforcement-learning)
4. [Resources & References](#resources--references)
    * [Academic Research](#academic-research)
    * [Industry News](#industry-news)
    * [Books](#books)
        - [Machine Learning](#machine-learning)
    * [Courses](#courses)
    * [ML Competitions & Trading](#ml-competitions--trading)
    * [Python Libraries](#python-libraries)

## The rise ka ML mein the investment industry

The investment industry has evolved dramatically over the last several decades aur continues to do so amid increased competition, technological advances, aur a challenging economic environment. Yeh section reviews key trends that have shaped the overall investment environment overall aur the context ke liye algorithmic trading aur the use ka ML more specifically.

The trends that have propelled algorithmic trading aur ML to current prominence include:
- Changes mein the market microstructure, such as the spread ka electronic trading aur the integration ka markets across asset classes aur geographies
- The development ka investment strategies framed mein terms ka risk-factor exposure, as opposed to asset classes
- The revolutions mein computing power, data generation aur management, aur statistical methods, including breakthroughs mein deep learning
- The outperformance ka the pioneers mein algorithmic trading relative to human, discretionary investors

mein addition, the financial crises ka 2001 aur 2008 have affected how investors approach diversification aur risk management. One outcome hai the rise to low-cost passive investment vehicles mein the form ka exchange-traded funds (ETFs). Amid low yields aur low volatility following the 2008 crisis that triggered large-scale asset purchases by leading central banks, cost-conscious investors shifted over $3.5 trillion from actively managed mutual funds into passively managed ETFs. 

Competitive pressure hai also reflected mein lower hedge fund fees that dropped from the traditional 2 percent annual management fee aur 20 percent take ka profits to an average ka 1.48 percent aur 17.4 percent, respectively, mein 2017.

### From electronic to high-frequency trading

Electronic trading has advanced dramatically mein terms ka capabilities, volume, coverage ka asset classes, aur geographies since networks started routing prices to computer terminals mein the 1960s.

- [Dark Pool Trading & Finance](https://www.cfainstitute.org/en/advocacy/issues/dark-pools), CFA Institute
- [Dark Pools mein Equity Trading: Policy Concerns aur Recent Developments](https://crsreports.congress.gov/product/pdf/R/R43739), Congressional Research Service, 2014
- [High Frequency Trading: Overview ka Recent Developments](https://fas.org/sgp/crs/misc/R44443.pdf), Congressional Research Service, 2016

### Factor investing aur smart beta funds

The return provided by an asset hai a function ka the uncertainty or risk associated ke saath the financial investment. An equity investment implies, ke liye example, assuming a company's business risk, aur a bond investment implies assuming default risk.

To the extent that specific risk characteristics predict returns, identifying aur forecasting the behavior ka these risk factors becomes a primary focus when designing an investment strategy. It yields valuable trading signals aur hai the key to superior active-management results. The industry's understanding ka risk factors has evolved very substantially over time aur has impacted how ML hai used ke liye algorithmic trading.

The factors that explained returns above aur beyond the CAPM were incorporated into investment styles that tilt portfolios mein favor ka one or more factors, aur assets began to migrate into factor-based portfolios. The 2008 financial crisis underlined how asset-class labels could be highly misleading aur create a false sense ka diversification when investors do not look at the underlying factor risks, as asset classes came crashing down together.

Over the past several decades, quantitative factor investing has evolved from a simple approach based on two or three styles to multifactor smart or exotic beta products. Smart beta funds have crossed $1 trillion AUM mein 2017, testifying to the popularity ka the hybrid investment strategy that combines active aur passive management. Smart beta funds take a passive strategy but modify it according to one or more factors, such as cheaper stocks or screening them according to dividend payouts, to generate better returns. Yeh growth has coincided ke saath increasing criticism ka the high fees charged by traditional active managers as well as heightened scrutiny ka their performance.

The ongoing discovery aur successful forecasting ka risk factors that, either individually or mein combination ke saath other risk factors, significantly impact future asset returns across asset classes hai a key driver ka the surge mein ML mein the investment industry aur will be a key theme throughout this book.

### Algorithmic pioneers outperform humans

The track record aur growth ka Assets Under Management (AUM) ka firms that spearheaded algorithmic trading has played a key role mein generating investor interest aur subsequent industry efforts to replicate their success.

Systematic strategies that mostly or exclusively rely on algorithmic decision-making were most famously introduced by mathematician James Simons who founded Renaissance Technologies mein 1982 aur built it into the premier quant firm. Its secretive Medallion Fund, which hai closed to outsiders, has earned an estimated annualized return ka 35% since 1982.

DE Shaw, Citadel, aur Two Sigma, three ka the most prominent quantitative hedge funds that use systematic strategies based on algorithms, rose to the all-time top-20 performers ke liye the first time mein 2017 mein terms ka total dollars earned ke liye investors, after fees, aur since inception.

#### ML driven funds attract $1 trillion AUM

Morgan Stanley estimated mein 2017 that algorithmic strategies have grown at 15% per year over the past six years aur control about $1.5 trillion between hedge funds, mutual funds, aur smart beta ETFs. Other reports suggest the quantitative hedge fund industry was about to exceed $1 trillion AUM, nearly doubling its size since 2010 amid outflows from traditional hedge funds. mein contrast, total hedge fund industry capital hit $3.21 trillion according to the latest global Hedge Fund Research report.

- [Global Algorithmic Trading Market to Surpass US$ 21,685.53 Million by 2026](https://www.bloomberg.com/press-releases/2019-02-05/global-algorithmic-trading-market-to-surpass-us-21-685-53-million-by-2026)
- [The stockmarket hai now run by computers, algorithms aur passive managers](https://www.economist.com/briefing/2019/10/05/the-stockmarket-hai-now-run-by-computers-algorithms-aur-passive-managers), Economist, Oct 5, 2019

#### The emergence ka quantamental funds

Two distinct approaches have evolved mein active investment management: systematic (or quant) aur discretionary investing. Systematic approaches rely on algorithms ke liye a repeatable aur data-driven approach to identify investment opportunities across many securities; mein contrast, a discretionary approach involves an mein-depth analysis ka a smaller number ka securities. These two approaches hain becoming more similar as fundamental managers take more data-science-driven approaches.

Even fundamental traders now arm themselves ke saath quantitative techniques, accounting ke liye $55 billion ka systematic assets, according to Barclays. Agnostic to specific companies, quantitative funds trade patterns aur dynamics across a wide swath ka securities. Quants now account ke liye about 17% ka total hedge fund assets, data compiled by Barclays shows.

### ML aur alternative data

Hedge funds have long looked ke liye alpha through informational advantage aur the ability to uncover new uncorrelated signals. Historically, this included things such as proprietary surveys ka shoppers, or voters ahead ka elections or referendums. Occasionally, the use ka company insiders, doctors, aur expert networks to expand knowledge ka industry trends or companies crosses legal lines: a series ka prosecutions ka traders, portfolio managers, aur analysts ke liye use karke insider information after 2010 has shaken the industry.

mein contrast, the informational advantage from exploiting conventional aur alternative data sources use karke ML hai not related to expert aur industry networks or access to corporate management, but rather the ability to collect large quantities ka data aur analyze them mein real-time.

Three trends have revolutionized the use ka data mein algorithmic trading strategies aur may further shift the investment industry from discretionary to quantitative styles:
- The exponential increase mein the amount ka digital data 
- The increase mein computing power aur data storage capacity at lower cost
- The advances mein ML methods ke liye analyzing complex datasets

- [Can hum Predict the Financial Markets Based on Google's Search Queries?](https://onlinelibrary.wiley.com/doi/abs/10.1002/ke liye.2446), Perlin, et al, 2016, Journal ka Forecasting

## Designing aur executing an ML-driven strategy

ML can add value at multiple steps mein the lifecycle ka a trading strategy, aur relies on key infrastructure aur data resources. Hence, this book aims to addresses how ML techniques fit into the broader process ka designing, executing, aur evaluating strategies.

An algorithmic trading strategy hai driven by a combination ka alpha factors that transform one or several data sources into signals that mein turn predict future asset returns aur trigger buy or sell orders. Chapter 2, Market aur Fundamental Data aur Chapter 3, Alternative Data ke liye Finance cover the sourcing aur management ka data, the raw material aur the single most important driver ka a successful trading strategy.  

[Chapter 4, Alpha Factor Research](../04_alpha_factor_research) outlines a methodologically sound process to manage the risk ka false discoveries that increases ke saath the amount ka data. [Chapter 5, Strategy Evaluation](../05_strategy_evaluation) provide karta hai the context ke liye the execution aur performance measurement ka a trading strategy.

The following subsections outline these steps, which hum will discuss mein depth throughout the book.

### Sourcing aur managing data

The dramatic evolution ka data availability mein terms ka volume, variety, aur velocity hai a key complement to the application ka ML to trading, which mein turn has boosted industry spending on the acquisition ka new data sources. However, the proliferating supply ka data requires careful selection aur management to uncover the potential value, including the following steps:

1. Identify aur evaluate market, fundamental, aur alternative data sources containing alpha signals that do not decay too quickly.
2. Deploy or access a cloud-based scalable data infrastructure aur analytical tools like Hadoop or Spark to facilitate fast, flexible data access.
3. Carefully manage aur curate data to avoid look-ahead bias by adjusting it to the desired frequency on a point-mein-time basis. Yeh means that data should reflect only information available aur known at the given time. ML algorithms trained on distorted historical data will almost certainly fail during live trading.

hum will cover these aspects mein practical detail mein Chapter 2, Market aur Fundamental Data: Sources aur Techniques, aur Chapter 3, Alternative Data ke liye Finance: Categories aur Use Cases.

### From alpha factor research to portfolio management

Alpha factors hain designed to extract signals from data to predict asset returns ke liye a given investment universe over the trading horizon. A factor takes on a single value ke liye each asset when evaluated, but may combine one or several input variables. The process involves the steps outlined mein the following figure:

The Research phase ka the trading strategy workflow includes the design, evaluation, aur combination ka alpha factors. ML plays a large role mein this process because the complexity ka factors has increased as investors react to both the signal decay ka simpler factors aur the much richer data available today.

Alpha factors emit entry aur exit signals that lead to buy or sell orders, aur order execution results mein portfolio holdings. The risk profiles ka individual positions interact to create a specific portfolio risk profile. Portfolio management involves the optimization ka position weights to achieve the desired portfolio risk aur return a profile that aligns ke saath the overall investment objectives. Yeh process hai highly dynamic to incorporate continuously-evolving market data.

### Strategy backtesting

The incorporation ka an investment idea into an algorithmic strategy requires extensive testing ke saath a scientific approach that attempts to reject the idea based on its performance mein alternative out-ka-sample market scenarios. Testing may involve simulated data to capture scenarios deemed possible but not reflected mein historic data.

## ML ke liye trading mein practice: strategies aur use cases

mein practice, hum apply ML to trading mein the context ka a specific strategy to meet a certain business goal. mein this section, hum briefly describe how trading strategies have evolved aur diversified, aur outline real-world examples ka ML applications, highlighting how they relate to the content covered mein this book.

### The evolution ka algorithmic strategies

Quantitative strategies have evolved aur become more sophisticated mein three waves:

1. mein the 1980s aur 1990s, signals often emerged from academic research aur used a single or very few inputs derived from market aur fundamental data. AQR, one ka the largest quantitative hedge funds today, was founded mein 1998 to implement such strategies at scale. These signals hain now largely commoditized aur available as ETF, such as basic mean-reversion strategies.
2. mein the 2000s, factor-based investing proliferated based on the pioneering work by Eugene Fama aur Kenneth French aur others. Funds used algorithms to identify assets exposed to risk factors like value or momentum to seek arbitrage opportunities. Redemptions during the early days ka the financial crisis triggered the quant quake ka August 2007 that cascaded through the factor-based fund industry. These strategies hain now also available as long-only smart beta funds that tilt portfolios according to a given set ka risk factors.
3. The third era hai driven by investments mein ML capabilities aur alternative data to generate profitable signals ke liye repeatable trading strategies. Factor decay hai a major challenge: the excess returns from new anomalies have been shown to drop by a quarter from discovery to publication, aur by over 50 percent after publication due to competition aur crowding.

Today, traders pursue a range ka different objectives when use karke algorithms to execute rules:
- Trade execution algorithms that aim to achieve favorable pricing
- Short-term trades that aim to profit from small price movements, ke liye example, due to arbitrage
- Behavioral strategies that aim to anticipate the behavior ka other market participants
- Trading strategies based on absolute aur relative price aur return predictions

### Use cases ka ML ke liye trading

ML extracts signals from a wide range ka market, fundamental, aur alternative data, aur can be applied at all steps ka the algorithmic trading-strategy process. Key applications include:
- Data mining to identify patterns, extract features aur generate insights
- Supervised learning to generate risk factors or alphas aur create trade ideas
- Aggregation ka individual signals into a strategy
- Allocation ka assets according to risk profiles learned by an algorithm
- The testing aur evaluation ka strategies, including through the use ka synthetic data
- The interactive, automated refinement ka a strategy use karke reinforcement learning

hum briefly highlight some ka these applications aur identify where hum will demonstrate their use mein later chapters.

#### Data mining ke liye feature extraction aur insights

The cost-effective evaluation ka large, complex datasets requires the detection ka signals at scale. There hain several examples throughout the book:
- **Information theory** helps estimate a signal content ka candidate features hai thus useful ke liye extracting the most valuable inputs ke liye an ML model. mein Chapter 4, Financial Feature Engineering: How to Research Alpha Factors, hum use mutual information to compare the potential values ka individual features ke liye a supervised learning algorithm to predict asset returns. Chapter 18 mein De Prado (2018) estimates the information content ka a price series as a basis ke liye deciding between alternative trading strategies.
- **Unsupervised learning** provide karta hai a broad range ka methods to identify structure mein data to gain insights or help solve a downstream task. hum provide several examples: 
    - In Chapter 13, [Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity](../13_unsupervised_learning/README.md), we introduce clustering and dimensionality reduction to generate features from high-dimensional datasets. 
    - In Chapter 15, [Topic Modeling for Earnings Calls and Financial News](../15_topic_modeling/README.md), we apply Bayesian probability models to summarize financial text data.
    - In Chapter 20: [Autoencoders for Conditional Risk Factors](../20_autoencoders_for_conditional_risk_factors), we used deep learning to extract non-linear risk factors conditioned on asset characteristics and predict stock returns based on [Kelly et. al.](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) (2020).
- **Model transparency**: hum emphasize model-specific ways to gain insights into the predictive power ka individual variables aur introduce a novel game-theoretic approach called SHapley Additive exPlanations (SHAP). hum apply it to gradient boosting machines ke saath a large number ka input variables mein Chapter 12, Boosting your Trading Strategy aur the Appendix.

#### Supervised learning ke liye alpha factor creation aur aggregation

The most familiar rationale ke liye applying ML to trading hai to obtain predictions ka asset fundamentals, price movements, or market conditions. A strategy can leverage multiple ML algorithms that build on each other:

- **Downstream models** can generate signals at the portfolio level by integrating predictions about the prospects ka individual assets, capital market expectations, aur the correlation among securities. 
- Alternatively, ML predictions can inform **discretionary trades** as mein the quantamental approach outlined previously. 

ML predictions can also **target specific risk factors**, such as value or volatility, or implement technical approaches, such as trend-following or mean reversion:
- mein Chapter 3, [Alternative Data ke liye Finance: Categories aur Use Cases](../03_alternative_data/README.md), hum illustrate how to work ke saath fundamental data to create inputs to ML-driven valuation models.
- mein Chapter 14, [Text Data ke liye Trading: Sentiment Analysis](../14_working_with_text_data/README.md), Chapter 15, [Topic Modeling ke liye Earnings Calls aur Financial News](../15_topic_modeling/README.md), aur Chapter 16, [Extracting Better Features: Word Embeddings ke liye Earnings Calls aur SEC Filings](../16_word_embeddings/README.md), hum use alternative data on business reviews that can be used to project revenues ke liye a company as an input ke liye a valuation exercise.
- mein Chapter 9, [From Volatility Forecasts to Statistical Arbitrage: Time Series Models](../09_time_series_models/README.md), hum demonstrate how to forecast macro variables as inputs to market expectations aur how to forecast risk factors such as volatility
- mein Chapter 19, [RNNs ke liye Trading: Multivariate Return Series aur Text Data](../19_recurrent_neural_nets/README.md), hum introduce recurrent neural networks that achieve superior performance ke saath nonlinear time series data.

#### Asset allocation
ML has been used to allocate portfolios based on decision-tree models that compute a hierarchical form ka risk parity. As a result, risk characteristics hain driven by patterns mein asset prices rather than by asset classes aur achieve superior risk-return characteristics.

- mein Chapter 5, [Portfolio Optimization aur Performance Evaluation](../05_strategy_evaluation/README.md), aur Chapter 13, [Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity](../13_unsupervised_learning/README.md), hum illustrate how hierarchical clustering extracts data-driven risk classes that better reflect correlation patterns than conventional asset class definition (see Chapter 16 mein De Prado, 2018).

#### Testing trade ideas

Backtesting hai a critical step to select successful algorithmic trading strategies. Cross-validation use karke synthetic data hai a key ML technique to generate reliable out-ka-sample results when combined ke saath appropriate methods to correct ke liye multiple testing. The time-series nature ka financial data requires modifications to the standard approach to avoid look-ahead bias or otherwise contaminate the data used ke liye training, validation, aur testing. mein addition, the limited availability ka historical data has given rise to alternative approaches that use synthetic data:
hum will demonstrate various methods to test ML models use karke market, fundamental, aur alternative that obtain sound estimates ka out-ka-sample errors.
mein Chapter 21, [Generative Adversarial Networks ke liye Synthetic Training Data](../21_gans_for_synthetic_time_series/README.md), hum present generative adversarial networks (GANs) that hain capable ka producing high-quality synthetic data.

#### Reinforcement learning

Trading takes place mein a competitive, interactive marketplace. Reinforcement learning aims to train agents to learn a policy function based on rewards; it hai often considered as one ka the most promising areas mein financial ML. See, e.g. Hendricks aur Wilcox (2014) aur Nevmyvaka, Feng, aur Kearns (2006) ke liye applications to trade execution.
- mein Chapter 22, [Deep Reinforcement Learning: Building a Trading Agent](../22_deep_reinforcement_learning/README.md), hum present key reinforcement algorithms like Q-learning to demonstrate the training ka reinforcement algorithms ke liye trading use karke OpenAI's Gym environment.

## Sansadhan (Resources) & References

### Academic Research

- [The fundamental law ka active management](http://jpm.iijournals.com/content/15/3/30), Richard C. Grinold, The Journal ka Portfolio Management Spring 1989, 15 (3) 30-37
- [The relationship between return aur market value ka common stocks](https://www.sciencedirect.com/science/article/pii/0304405X81900180), Rolf Banz,Journal ka Financial Economics, March 1981
- [The Arbitrage Pricing Theory: Some Empirical Results](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1981.tb00444.x), Marc Reinganum, Journal ka Finance, 1981
- [The Relationship between Earnings' Yield, Market Value aur Return ke liye NYSE Common Stock](https://pdfs.semanticscholar.org/26ab/311756099c8f8c4e528083c9b90ff154f98e.pdf), Sanjoy Basu, Journal ka Financial Economics, 1982
- [Bridging the divide mein financial market forecasting: machine learners vs. financial economists](http://www.sciencedirect.com/science/article/pii/S0957417416302585), Expert Systems ke saath Applications, 2016 
- [Financial Time Series Forecasting ke saath Deep Learning : A Systematic Literature Review: 2005-2019](http://arxiv.org/abs/1911.13288), arXiv:1911.13288 [cs, q-fin, stat], 2019 
- [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009), The Review ka Financial Studies, 2020 
- [The Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns](http://academic.oup.com/rfs/article/30/12/4389/3091648), The Review ka Financial Studies, 2017 
- [Characteristics hain covariances: A unified model ka risk aur return](http://www.sciencedirect.com/science/article/pii/S0304405X19301151), Journal ka Financial Economics, 2019 
- [Estimation aur Inference ka Heterogeneous Treatment Effects use karke Random Forests](https://doi.org/10.1080/01621459.2017.1319839), Journal ka the American Statistical Association, 2018 
- [An Empirical Study ka Machine Learning Algorithms ke liye Stock Daily Trading Strategy](https://www.hindawi.com/journals/mpe/2019/7816154/), Mathematical Problems mein Engineering, 2019 
- [Predicting stock market index use karke fusion ka machine learning techniques](http://www.sciencedirect.com/science/article/pii/S0957417414006551), Expert Systems ke saath Applications, 2015 
- [Predicting stock aur stock price index movement use karke Trend Deterministic Data Preparation aur machine learning techniques](http://www.sciencedirect.com/science/article/pii/S0957417414004473), Expert Systems ke saath Applications, 2015 
- [Deep Learning ke liye Limit Order Books](http://arxiv.org/abs/1601.01987), arXiv:1601.01987 [q-fin], 2016 
- [Trading via Image Classification](http://arxiv.org/abs/1907.10046), arXiv:1907.10046 [cs, q-fin], 2019 
- [Algorithmic trading review](http://doi.org/10.1145/2500117), Communications ka the ACM, 2013 
- [Assessing the impact ka algorithmic trading on markets: A simulation approach](https://www.econstor.eu/handle/10419/43250), , 2008 
- [The Efficient Market Hypothesis aur Its Critics](http://www.aeaweb.org/articles?id=10.1257/089533003321164958), Journal ka Economic Perspectives, 2003 
- [The Arbitrage Pricing Theory Approach to Strategic Portfolio Planning](https://doi.org/10.2469/faj.v40.n3.14), Financial Analysts Journal, 1984 

### Industry News

- [The Rise ka the Artificially Intelligent Hedge Fund](https://www.wired.com/2016/01/the-rise-ka-the-artificially-intelligent-hedge-fund/#comments), Wired, 25-01-2016
- [Crowd-Sourced Quant Network Allocates Most Ever to Single Algo](https://www.bloomberg.com/news/articles/2018-08-02/crowd-sourced-quant-network-allocates-most-ever-to-single-algo), Bloomberg, 08-02-2018
- [Goldman Sachs’ lessons from the ‘quant quake’](https://www.ft.com/content/fdfd5e78-0283-11e7-aa5b-6bb07f5c8e12), Financial Times, 03-08-2017
- [Lessons from the Quant Quake resonate a decade later](https://www.ft.com/content/a7a04d4c-83ed-11e7-94e2-c5b903247afd), Financial Times, 08-18-2017
- [Smart beta funds pass $1tn mein assets](https://www.ft.com/content/bb0d1830-e56b-11e7-8b99-0191e45377ec), Financial Times, 12-27-2017
- [BlackRock bets on algorithms to beat the fund managers](https://www.ft.com/content/e689a67e-2911-11e8-b27e-cc62a39d57a0), Financial Times, 03-20-2018
- [Smart beta: what’s mein a name?](https://www.ft.com/content/d1bdabaa-a9f0-11e7-ab66-21cc87a2edde), Financial Times, 11-27-2017
- [Computer-driven hedge funds join industry top performers](https://www.ft.com/content/9981c870-e79a-11e6-967b-c88452263daf), Financial Times, 02-01-2017
- [Quants Rule Alpha’s Hedge Fund 100 List](https://www.institutionalinvestor.com/article/b1505pmf2v2hg3/quants-rule-alphas-hedge-fund-100-list), Institutional Investor, 06-26-2017
- [The Quants Run Wall Street Now](https://www.wsj.com/articles/the-quants-run-wall-street-now-1495389108), Wall Street Journal, 05-21-2017
- ['hum Don’t Hire MBAs': The New Hedge Fund Winners Will Crunch The Better Data Sets](https://www.cbinsights.com/research/algorithmic-hedge-fund-trading-winners/), cbinsights, 06-28-2018
- [Artificial Intelligence: Fusing Technology aur Human Judgment?](https://blogs.cfainstitute.org/investor/2017/09/25/artificial-intelligence-fusing-technology-aur-human-judgment/), CFA Institute, 09-25-2017
- [The Hot New Hedge Fund Flavor hai 'Quantamental'](https://www.bloomberg.com/news/articles/2017-08-25/the-hot-new-hedge-fund-flavor-hai-quantamental-quicktake-q-a), Bloomberg, 08-25-2017
- [Robots hain Eating Money Managers’ Lunch](https://www.bloomberg.com/news/articles/2017-06-20/robots-hain-eating-money-managers-lunch), Bloomberg, 06-20-2017
- [Rise ka Robots: Inside the World's Fastest Growing Hedge Funds](https://www.bloomberg.com/news/articles/2017-06-20/rise-ka-robots-inside-the-world-s-fastest-growing-hedge-funds), Bloomberg, 06-20-2017
- [When Silicon Valley came to Wall Street](https://www.ft.com/content/ba5dc7ca-b3ef-11e7-aa26-bb002965bce8), Financial Times, 10-28-2017
- [BlackRock bulks up research into artificial intelligence](https://www.ft.com/content/4f5720ce-1552-11e8-9376-4a6390addb44), Financial Times, 02-19-2018
- [AQR to explore use ka ‘big data’ despite past doubts](https://www.ft.com/content/3a8f69f2-df34-11e7-a8a4-0a1e63a52f9c), Financial Times, 12-12-2017
- [Two Sigma rapidly rises to top ka quant hedge fund world](https://www.ft.com/content/dcf8077c-b823-11e7-9bfb-4a9c83ffa852), Financial Times, 10-24-2017
- [When Silicon Valley came to Wall Street](https://www.ft.com/content/ba5dc7ca-b3ef-11e7-aa26-bb002965bce8), Financial Times, 10-28-2017
- [Artificial intelligence (AI) mein finance - six warnings from a central banker](https://www.bundesbank.de/en/press/speeches/artificial-intelligence--ai--mein-finance--six-warnings-from-a-central-banker-711602), Deutsche Bundesbank, 02-27-2018
- [Fintech: Search ke liye a super-algo](https://www.ft.com/content/5eb91614-bee5-11e5-846f-79b0e3d20eaf), Financial Times, 01-20-2016
- [Barron’s Top 100 Hedge Funds](https://www.barrons.com/articles/top-100-hedge-funds-1524873705)
- [How high-frequency trading hit a speed bump](https://www.ft.com/content/d81f96ea-d43c-11e7-a303-9060cb1e5f44), FT, 01-01-2018

### Books

- [Advances mein Financial Machine Learning](https://www.wiley.com/en-us/Advances+mein+Financial+Machine+Learning-p-9781119482086), Marcos Lopez de Prado, 2018
- [Quantresearch](http://www.quantresearch.info/index.html) by Marcos López de Prado
- [Quantitative Trading](http://epchan.blogspot.com/), Ernest Chan
- [Machine Learning mein Finance](https://www.springer.com/gp/book/9783030410674), Dixon, Matthew F., Halperin, Igor, Bilokon, Paul, Springer, 2020

#### Machine Learning

- [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html), Tom Mitchell, McGraw Hill, 1997
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), Gareth James et al.
    - Excellent reference for essential machine learning concepts, available free online
- [Bayesian Reasoning aur Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf), Barber, D., Cambridge University Press, 2012 (updated version available on author's website)

### Courses

- [Algorithmic Trading](http://personal.stevens.edu/~syang14/fe670.htm), Prof. Steve Yang, Stevens Institute ka Technology
- [Machine Learning](https://www.coursera.org/learn/machine-learning), Andrew Ng, Coursera
- [Deep Learning Specialization](http://deeplearning.ai/), Andrew Ng
    - Andrew Ng’s introductory deep learning course
- Machine Learning ke liye Trading Specialization, [Coursera](https://www.coursera.org/specializations/machine-learning-trading)
- Machine Learning ke liye Trading, Georgia Tech CS 7646, [Udacity](https://www.udacity.com/course/machine-learning-ke liye-trading--ud501
- Introduction to Machine Learning ke liye Trading, [Quantinsti](https://quantra.quantinsti.com/course/introduction-to-machine-learning-ke liye-trading)

### ML Competitions & Trading

- [IEEE Investment Ranking Challenge](https://www.crowdai.org/challenges/ieee-investment-ranking-challenge)
    - [Investment Ranking Challenge : Identifying the best performing stocks based on their semi-annual returns](https://arxiv.org/pdf/1906.08636.pdf)
- [Two Sigma Financial Modeling Challenge](https://www.kaggle.com/c/two-sigma-financial-modeling)
- [Two Sigma: use karke News to Predict Stock Movements](https://www.kaggle.com/c/two-sigma-financial-news)
- [The Winton Stock Market Challenge](https://www.kaggle.com/c/the-winton-stock-market-challenge)
- [Algorithmic Trading Challenge](https://www.kaggle.com/c/AlgorithmicTradingChallenge)
   
### Python Libraries

- matplotlib [docs](https://github.com/matplotlib/matplotlib)
- numpy [docs](https://github.com/numpy/numpy)
- pandas [docs](https://github.com/pydata/pandas)
- scipy [docs](https://github.com/scipy/scipy)
- scikit-learn [docs](https://scikit-learn.org/stable/user_guide.html)
- LightGBM [docs](https://lightgbm.readthedocs.io/en/latest/)
- CatBoost [docs](https://catboost.ai/docs/concepts/about.html)
- TensorFlow [docs](https://www.tensorflow.org/guide)
- PyTorch [docs](https://pytorch.org/docs/stable/index.html)
- Machine Learning Financial Laboratory (mlfinlab) [docs](https://mlfinlab.readthedocs.io/en/latest/)
- seaborn [docs](https://github.com/mwaskom/seaborn)
- statsmodels [docs](https://github.com/statsmodels/statsmodels)
- [Boosting numpy: Why BLAS Matters](http://markus-beuckelmann.de/blog/boosting-numpy-blas.html)



















































