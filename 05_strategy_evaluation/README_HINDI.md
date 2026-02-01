# Portfolio Optimization aur Performance Evaluation

To test a strategy prior to implementation under market conditions, hum need to simulate the trades that the algorithm would make aur verify their performance. Strategy evaluation includes backtesting against historical data to optimize the strategy's parameters aur forward-testing to validate the mein-sample performance against new, out-ka-sample data. The goal hai to avoid false discoveries from tailoring a strategy to specific past circumstances.

mein a portfolio context, positive asset returns can offset negative price movements. Positive price changes ke liye one asset hain more likely to offset losses on another the lower the correlation between the two positions. Based on how portfolio risk depends on the positions’ covariance, Harry Markowitz developed the theory behind modern portfolio management based on diversification mein 1952. The result hai mean-variance optimization that selects weights ke liye a given set ka assets to minimize risk, measured as the standard deviation ka returns ke liye a given expected return.

The capital asset pricing model (CAPM) introduces a risk premium, measured as the expected return mein excess ka a risk-free investment, as an equilibrium reward ke liye holding an asset. Yeh reward compensates ke liye the exposure to a single risk factor—the market—that hai systematic as opposed to idiosyncratic to the asset aur thus cannot be diversified away. 

Risk management has evolved to become more sophisticated as additional risk factors aur more granular choices ke liye exposure have emerged. The Kelly criterion hai a popular approach to dynamic portfolio optimization, which hai the choice ka a sequence ka positions over time; it has been famously adapted from its original application mein gambling to the stock market by Edward Thorp mein 1968.

As a result, there hain several approaches to optimize portfolios that include the application ka machine learning (ML) to learn hierarchical relationships among assets aur treat their holdings as complements or substitutes ke saath respect to the portfolio risk profile. Yeh chapter will cover the following topics:

## Vishay-suchi (Content)

1. [How to measure portfolio performance](#how-to-measure-portfolio-performance)
    * [The (adjusted) Sharpe Ratio](#the-adjusted-sharpe-ratio)
    * [The fundamental law of active management](#the-fundamental-law-of-active-management)
2. [How to manage Portfolio Risk & Return](#how-to-manage-portfolio-risk--return)
    * [The evolution of modern portfolio management](#the-evolution-of-modern-portfolio-management)
    * [Mean-variance optimization](#mean-variance-optimization)
        - [Code Examples: Finding the efficient frontier in Python](#code-examples-finding-the-efficient-frontier-in-python)
    * [Alternatives to mean-variance optimization](#alternatives-to-mean-variance-optimization)
        - [The 1/N portfolio](#the-1n-portfolio)
        - [The minimum-variance portfolio](#the-minimum-variance-portfolio)
        - [The Black-Litterman approach](#the-black-litterman-approach)
        - [How to size your bets – the Kelly rule](#how-to-size-your-bets--the-kelly-rule)
        - [Alternatives to MV Optimization with Python](#alternatives-to-mv-optimization-with-python)
    * [Hierarchical Risk Parity](#hierarchical-risk-parity)
3. [Trading aur managing a portfolio ke saath `Zipline`](#trading-aur-managing-a-portfolio-ke saath-zipline)
    * [Code Examples: Backtests with trades and portfolio optimization ](#code-examples-backtests-with-trades-and-portfolio-optimization-)
4. [Measure backtest performance ke saath `pyfolio`](#measure-backtest-performance-ke saath-pyfolio)
    * [Code Example: `pyfolio` evaluation from a `Zipline` backtest](#code-example-pyfolio-evaluation-from-a-zipline-backtest)

## How to measure portfolio performance

To evaluate aur compare different strategies or to improve an existing strategy, hum need metrics that reflect their performance ke saath respect to our objectives. mein investment aur trading, the most common objectives hain the **return aur the risk ka the investment portfolio**.

The return aur risk objectives imply a trade-off: taking more risk may yield higher returns mein some circumstances, but also implies greater downside. To compare how different strategies navigate this trade-off, ratios that compute a measure ka return per unit ka risk hain very popular. hum’ll discuss the **Sharpe ratio** aur the **information ratio** (IR) mein turn.

### The (adjusted) Sharpe Ratio

The ex-ante Sharpe Ratio (SR) compares the portfolio's expected excess portfolio to the volatility ka this excess return, measured by its standard deviation. It measures the compensation as the average excess return per unit ka risk taken. It can be estimated from data.

Financial returns often violate the iid assumptions. Andrew Lo has derived the necessary adjustments to the distribution aur the time aggregation ke liye returns that hain stationary but autocorrelated. Yeh hai important because the time-series properties ka investment strategies (ke liye example, mean reversion, momentum, aur other forms ka serial correlation) can have a non-trivial impact on the SR estimator itself, especially when annualizing the SR from higher-frequency data.

- [The Statistics ka Sharpe Ratios](https://www.jstor.org/stable/4480405?seq=1#page_scan_tab_contents), Andrew Lo, Financial Analysts Journal, 2002

### The fundamental law ka active management

It’s a curious fact that Renaissance Technologies (RenTec), the top-performing quant fund founded by Jim Simons that hum mentioned mein [Chapter 1](../01_machine_learning_for_trading), has produced similar returns as Warren Buffet despite extremely different approaches. Warren Buffet’s investment firm Berkshire Hathaway holds some 100-150 stocks ke liye fairly long periods, whereas RenTec may execute 100,000 trades per day. How can hum compare these distinct strategies?

ML hai about optimizing objective functions. mein algorithmic trading, the objectives hain the return aur the risk ka the overall investment portfolio, typically relative to a benchmark (which may be cash, the risk-free interest rate, or an asset price index like the S&P 500).

A high Information Ratio (IR) implies attractive out-performance relative to the additional risk taken. The Fundamental Law ka Active Management breaks the IR down into the information coefficient (IC) as a measure ka forecasting skill, aur the ability to apply this skill through independent bets. It summarizes the importance to play both often (high breadth) aur to play well (high IC).

The IC measures the correlation between an alpha factor aur the forward returns resulting from its signals aur captures the accuracy ka a manager's forecasting skills. The breadth ka the strategy hai measured by the independent number ka bets an investor makes mein a given time period, aur the product ka both values hai proportional to the IR, also known as appraisal risk (Treynor aur Black).

The fundamental law hai important because it highlights the key drivers ka outperformance: both accurate predictions aur the ability to make independent forecasts aur act on these forecasts matter. mein practice, estimating the breadth ka a strategy hai difficult given the cross-sectional aur time-series correlation among forecasts. 

- [Active Portfolio Management: A Quantitative Approach ke liye Producing Superior Returns aur Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold aur Ronald Kahn, 1999
- [How to Use Security Analysis to Improve Portfolio Selection](https://econpapers.repec.org/article/ucpjnlbus/v_3a46_3ay_3a1973_3ai_3a1_3ap_3a66-86.htm), Jack L Treynor aur Fischer Black, Journal ka Business, 1973
- [Portfolio Constraints aur the Fundamental Law ka Active Management](https://faculty.fuqua.duke.edu/~charvey/Teaching/BA491_2005/Transfer_coefficient.pdf), Clarke et al 2002

## How to manage Portfolio Risk & Return

Portfolio management aims to pick aur size positions mein financial instruments that achieve the desired risk-return trade-off regarding a benchmark. As a portfolio manager, mein each period, you select positions that optimize diversification to reduce risks while achieving a target return. Across periods, these positions may require rebalancing to account ke liye changes mein weights resulting from price movements to achieve or maintain a target risk profile.

### The evolution ka modern portfolio management

Diversification permits us to reduce risks ke liye a given expected return by exploiting how imperfect correlation allows ke liye one asset's gains to make up ke liye another asset's losses. Harry Markowitz invented modern portfolio theory (MPT) mein 1952 aur provided the mathematical tools to optimize diversification by choosing appropriate portfolio weights.
 
### Mean-variance optimization

Modern portfolio theory solves ke liye the optimal portfolio weights to minimize volatility ke liye a given expected return, or maximize returns ke liye a given level ka volatility. The key requisite inputs hain expected asset returns, standard deviations, aur the covariance matrix. 
- [Portfolio Selection](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf), Harry Markowitz, The Journal ka Finance, 1952
- [The Capital Asset Pricing Model: Theory aur Evidence](http://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S6B-FamaFrench-CAPM-JEP04.pdf), Eugene F. Fama aur Kenneth R. French, Journal ka Economic Perspectives, 2004

#### Code Examples: Finding the efficient frontier mein Python

hum can calculate an efficient frontier use karke scipy.optimize.minimize aur the historical estimates ke liye asset returns, standard deviations, aur the covariance matrix. 
- Notebook [mean_variance_optimization](04_mean_variance_optimization.ipynb) to compute the efficient frontier mein python.

### Alternatives to mean-variance optimization

The challenges ke saath accurate inputs ke liye the mean-variance optimization problem have led to the adoption ka several practical alternatives that constrain the mean, the variance, or both, or omit return estimates that hain more challenging, such as the risk parity approach that hum discuss later mein this section.

#### The 1/N portfolio

Simple portfolios provide useful benchmarks to gauge the added value ka complex models that generate the risk ka overfitting. The simplest strategy—an equally-weighted portfolio—has been shown to be one ka the best performers.

#### The minimum-variance portfolio

Another alternative hai the global minimum-variance (GMV) portfolio, which prioritizes the minimization ka risk. It hai shown mein the efficient frontier figure aur can be calculated as follows by minimizing the portfolio standard deviation use karke the mean-variance framework.

#### The Black-Litterman approach

The Global Portfolio Optimization approach ka Black aur Litterman (1992) combines economic models ke saath statistical learning aur hai popular because it generates estimates ka expected returns that hain plausible mein many situations.
The technique assumes that the market hai a mean-variance portfolio as implied by the CAPM equilibrium model. It builds on the fact that the observed market capitalization can be considered as optimal weights assigned to each security by the market. Market weights reflect market prices that, mein turn, embody the market’s expectations ka future returns.

- [Global Portfolio Optimization](http://www.sef.hku.hk/tpg/econ6017/2011/black-litterman-1992.pdf), Black, Fischer; Litterman, Robert
Financial Analysts Journal, 1992

#### How to size your bets – the Kelly rule

The Kelly rule has a long history mein gambling because it provide karta hai guidance on how much to stake on each ka an (infinite) sequence ka bets ke saath varying (but favorable) odds to maximize terminal wealth. It was published as A New Interpretation ka the Information Rate mein 1956 by John Kelly who was a colleague ka Claude Shannon's at Bell Labs. He was intrigued by bets placed on candidates at the new quiz show The $64,000 Question, where a viewer on the west coast used the three-hour delay to obtain insider information about the winners. 

Kelly drew a connection to Shannon's information theory to solve ke liye the bet that hai optimal ke liye long-term capital growth when the odds hain favorable, but uncertainty remains. His rule maximizes logarithmic wealth as a function ka the odds ka success ka each game, aur includes implicit bankruptcy protection since log(0) hai negative infinity so that a Kelly gambler would naturally avoid losing everything.

- [A New Interpretation ka Information Rate](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf), John Kelly, 1956
- [Beat the Dealer: A Winning Strategy ke liye the Game ka Twenty-One](https://www.amazon.com/Beat-Dealer-Winning-Strategy-Twenty-One/dp/0394703103), Edward O. Thorp,1966
- [Beat the Market: A Scientific Stock Market System](https://www.researchgate.net/publication/275756748_Beat_the_Market_A_Scientific_Stock_Market_System) , Edward O. Thorp,1967
- [Quantitative Trading: How to Build Your Own Algorithmic Trading Business](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/0470284889/ref=sr_1_2?s=books&ie=UTF8&qid=1545525861&sr=1-2), Ernie Chan, 2008

#### Alternatives to MV Optimization ke saath Python

- Notebook [kelly_rule](05_kelly_rule.ipynb) demonstrate karta hai the application ke liye the single aur multiple asset case. 
- The latter result hai also included mein the notebook [mean_variance_optimization](04_mean_variance_optimization.ipynb), along ke saath several other alternative approaches.

### Hierarchical Risk Parity

Yeh novel approach developed by [Marcos Lopez de Prado](http://www.quantresearch.org/) aims to address three major concerns ka quadratic optimizers, mein general, aur Markowitz’s critical line algorithm (CLA), mein particular: 
- instability, 
- concentration, aur 
- underperformance. 

Hierarchical Risk Parity (HRP) applies graph theory aur machine-learning to build a diversified portfolio based on the information contained mein the covariance matrix. However, unlike quadratic optimizers, HRP does not require the invertibility ka the covariance matrix. mein fact, HRP can compute a portfolio on an ill-degenerated or even a singular covariance matrix—an impossible feat ke liye quadratic optimizers. Monte Carlo experiments show that HRP delivers lower out-ka-sample variance than CLA, even though minimum variance hai CLA’s optimization objective. HRP also produces less risky portfolios out ka sample compared to traditional risk parity methods. hum will discuss HRP mein more detail mein [Chapter 13](../13_unsupervised_learning) when hum discuss applications ka unsupervised learning, including hierarchical clustering, to trading.

- [Building diversified portfolios that outperform out ka sample](https://jpm.pm-research.com/content/42/4/59.short), Marcos López de Prado, The Journal ka Portfolio Management 42, no. 4 (2016): 59-69.
- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Thomas Raffinot, 2016

hum demonstrate how to implement HRP aur compare it to alternatives mein Chapter 13 on [Unsupervised Learning](../13_unsupervised_learning) where hum also introduce hierarchical clustering.

## Trading aur managing a portfolio ke saath `Zipline`

The open source [zipline](https://zipline.ml4trading.io/index.html) library hai an event-driven backtesting system maintained aur used mein production by the crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) to facilitate algorithm-development aur live-trading. It automates the algorithm's reaction to trade events aur provide karta hai it ke saath current aur historical point-mein-time data that avoids look-ahead bias. [Chapter 8 - The ML4T Workflow](../08_strategy_workflow) has a more detailed, dedicated introduction to backtesting use karke both `zipline` aur `backtrader`. 

mein [Chapter 4](../04_alpha_factor_research), hum introduced `zipline` to simulate the computation ka alpha factors from trailing cross-sectional market, fundamental, aur alternative data. Now hum will exploit the alpha factors to derive aur act on buy aur sell signals. 

### Code Examples: Backtests ke saath trades aur portfolio optimization 

The code ke liye this section lives mein the following two notebooks: 
- Notebooks mein this section use the `conda` environment `backtest`. Please see the installation [instructions](../installation/README.md) ke liye downloading the latest Docker image or alternative ways to set up your environment.
- Notebook [backtest_with_trades](01_backtest_with_trades.ipynb) simulates the trading decisions that build a portfolio based on the simple MeanReversion alpha factor from the last chapter use karke Zipline. hum not explicitly optimize the portfolio weights aur just assign positions ka equal value to each holding.
- Notebook [backtest_with_pf_optimization](02_backtest_with_pf_optimization.ipynb) demonstrate karta hai how to use PF optimization as part ka a simple strategy backtest. 

## Measure backtest performance ke saath `pyfolio`

Pyfolio facilitates the analysis ka portfolio performance aur risk mein-sample aur out-ka-sample use karke many standard metrics. It produces tear sheets covering the analysis ka returns, positions, aur transactions, as well as event risk during periods ka market stress use karke several built-mein scenarios, aur also includes Bayesian out-ka-sample performance analysis.

### Code Example: `pyfolio` evaluation from a `Zipline` backtest

Notebook [pyfolio_demo](03_pyfolio_demo.ipynb) illustrates how to extract the `pyfolio` input from the backtest conducted mein the previous folder. It then proceeds to calculate several performance metrics aur tear sheets use karke `pyfolio`

- This notebook requires the `conda` environment `backtest`. Please see the [installation instructions](../installation/README.md) ke liye running the latest Docker image or alternative ways to set up your environment.