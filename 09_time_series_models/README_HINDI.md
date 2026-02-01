# From Volatility Forecasts to Statistical Arbitrage: Linear Time Series Models

mein this chapter, hum will build dynamic linear models to explicitly represent time aur include variables observed at specific intervals or lags. A key characteristic ka time-series data hai their sequential order: rather than random samples ka individual observations as mein the case ka cross-sectional data, our data hain a single realization ka a stochastic process that hum cannot repeat.

Our goal hai to identify **systematic patterns mein time series** that help us predict how the time series will behave mein the future. More specifically, hum focus on models that extract signals from a historical sequence ka the output aur, optionally, other contemporaneous or lagged input variables to predict future values ka the output. Udaharan ke liye, hum might try to predict future returns ke liye a stock use karke past returns, combined ke saath historical returns ka a benchmark or macroeconomic variables. hum focus on linear time-series models before turning to nonlinear models like recurrent or convolutional neural networks mein Part 4. 

Time-series models hain very popular given the time dimension inherent to trading. Key applications include the **prediction ka asset returns aur volatility**, as well as the identification ka co-movements ka asset price series. Time-series data hain likely to become more prevalent as an ever-broader array ka connected devices collects regular measurements ke saath potential signal content.

hum first introduce tools to diagnose time-series characteristics aur to extract features that capture potential patterns. Then hum introduce univariate aur multivariate time-series models aur apply them to forecast macro data aur volatility patterns. hum conclude ke saath the concept ka **cointegration** aur how to apply it to develop a **pairs trading strategy**.

## Vishay-suchi (Content)

1. [Tools ke liye diagnostics aur feature extraction](#tools-ke liye-diagnostics-aur-feature-extraction)
    * [How to decompose time series patterns](#how-to-decompose-time-series-patterns)
    * [Rolling window statistics and moving averages](#rolling-window-statistics-and-moving-averages)
    * [How to measure autocorrelation](#how-to-measure-autocorrelation)
    * [How to diagnose and achieve stationarity](#how-to-diagnose-and-achieve-stationarity)
    * [How to apply time series transformations](#how-to-apply-time-series-transformations)
    * [How to diagnose and address unit roots](#how-to-diagnose-and-address-unit-roots)
    * [Code example: working with time series data](#code-example-working-with-time-series-data)
    * [Resources](#resources)
2. [Univariate Time Series Models](#univariate-time-series-models)
    * [How to build autoregressive models](#how-to-build-autoregressive-models)
    * [How to build moving average models](#how-to-build-moving-average-models)
    * [How to build ARIMA models and extensions](#how-to-build-arima-models-and-extensions)
    * [Code example: forecasting macro fundamentals with ARIMA and SARIMAX models](#code-example-forecasting-macro-fundamentals-with-arima-and-sarimax-models)
    * [How to use time series models to forecast volatility](#how-to-use-time-series-models-to-forecast-volatility)
    * [How to build a volatility-forecasting model](#how-to-build-a-volatility-forecasting-model)
    * [Code examples: volatility forecasts](#code-examples-volatility-forecasts)
    * [Resources](#resources-2)
3. [Multivariate Time Series Models](#multivariate-time-series-models)
    * [The vector autoregressive (VAR) model](#the-vector-autoregressive-var-model)
    * [Code example: How to use the VAR model for macro fundamentals forecasts](#code-example-how-to-use-the-var-model-for-macro-fundamentals-forecasts)
    * [Resources](#resources-3)
4. [Cointegration – time series ke saath a common trend](#cointegration--time-series-ke saath-a-common-trend)
    * [Pairs trading: Statistical arbitrage with cointegration](#pairs-trading-statistical-arbitrage-with-cointegration)
    * [Alternative approaches to selecting and trading comoving assets](#alternative-approaches-to-selecting-and-trading-comoving-assets)
    * [Code example: Pairs trading in practice](#code-example-pairs-trading-in-practice)
        - [Computing distance-based heuristics to identify cointegrated pairs](#computing-distance-based-heuristics-to-identify-cointegrated-pairs)
        - [Precomputing the cointegration tests](#precomputing-the-cointegration-tests)
    * [Resources](#resources-4)

## Tools ke liye diagnostics aur feature extraction

Most ka the examples mein this section use data provided by the Federal Reserve that you can access use karke the pandas datareader that hum introduced mein [Chapter 2, Market aur Fundamental Data](../02_market_and_fundamental_data). 

### How to decompose time series patterns

Time series data typically contain karta hai a mix ka various patterns that can be decomposed into several components, each representing an underlying pattern category. mein particular, time series often consist ka the systematic components trend, seasonality aur cycles, aur unsystematic noise. These components can be combined mein an additive, linear model, mein particular when fluctuations do not depend on the level ka the series, or mein a non-linear, multiplicative model. 

- `pandas` Time Series aur Date functionality [docs](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)
- [Forecasting - Principles & Practice, Hyndman, R. aur Athanasopoulos, G., ch.6 'Time Series Decomposition'](https://otexts.org/fpp2/decomposition.html)

### Rolling window statistics aur moving averages

The pandas library includes very flexible functionality to define various window types, including rolling, exponentially weighted aur expanding windows.

- `pandas` window function [docs](https://pandas.pydata.org/pandas-docs/stable/computation.html#window-functions)

### How to measure autocorrelation

Autocorrelation (also called serial correlation) adapts the concept ka correlation to the time series context: just as the correlation coefficient measures the strength ka a linear relationship between two variables, the autocorrelation coefficient measures the extent ka a linear relationship between time series values separated by a given lag.

hum present the following tools to measure autocorrelation:
- autocorrelation function (ACF)
- partial autocorrelation function (PACF)
- correlogram as a plot ka ACF or PACF against the number ka lags.

### How to diagnose aur achieve stationarity

The statistical properties, such as the mean, variance, or autocorrelation, ka a stationary time series hain independent ka the period, that hai, they don't change over time. Hence, stationarity implies that a time series does not have a trend or seasonal effects aur that descriptive statistics, such as the mean or the standard deviation, when computed ke liye different rolling windows, hain constant or do not change much over time.

### How to apply time series transformations

To satisfy the stationarity assumption ka linear time series models, hum need to transform the original time series, often mein several steps. Common transformations include the application ka the (natural) logarithm to convert an exponential growth pattern into a linear trend aur stabilize the variance, or differencing.

### How to diagnose aur address unit roots

Unit roots pose a particular problem ke liye determining the transformation that will render a time series stationary. mein practice, time series ka interest rates or asset prices hain often not stationary, ke liye example, because there does not exist a price level to which the series reverts. The most prominent example ka a non-stationary series hai the random walk.

The defining characteristic ka a unit-root non-stationary series hai long memory: since current values hain the sum ka past disturbances, large innovations persist ke liye much longer than ke liye a mean-reverting, stationary series. Identifying the correct transformation, aur mein particular, the appropriate number aur lags ke liye differencing hai not always clear-cut. hum present a few heuristics to guide the process.

Statistical unit root tests hain a common way to determine objectively whether (additional) differencing hai necessary. These hain statistical hypothesis tests ka stationarity that hain designed to determine whether differencing hai required.

### Code example: working ke saath time series data

- Notebook [tsa_and_stationarity](01_tsa_and_stationarity.ipynb) illustrates the concepts discussed mein this section.

### Sansadhan (Resources)

- [Analysis ka Financial Time Series, 3rd Edition, Ruey S. Tsay](https://www.wiley.com/en-us/Analysis+ka+Financial+Time+Series%2C+3rd+Edition-p-9780470414354)
- [Quantitative Equity Investing: Techniques aur Strategies, Frank J. Fabozzi, Sergio M. Focardi, Petter N. Kolm](https://www.wiley.com/en-us/Quantitative+Equity+Investing%3A+Techniques+aur+Strategies-p-9780470262474)
- `statsmodels` Time Series Analysis [docs](https://www.statsmodels.org/dev/tsa.html)

## Univariate Time Series Models

Univariate time series models relate the value ka the time series at the point mein time ka interest to a linear combination ka lagged values ka the series aur possibly past disturbance terms.

While exponential smoothing models hain based on a description ka the trend aur seasonality mein the data, ARIMA models aim to describe the autocorrelations mein the data. ARIMA(p, d, q) models require stationarity aur leverage two building blocks:
- Autoregressive (AR) terms consisting ka p-lagged values ka the time series
- Moving average (MA) terms that contain q-lagged disturbances

### How to build autoregressive models

An AR model ka order p aims to capture the linear dependence between time series values at different lags. It closely resembles a multiple linear regression on lagged values ka the outcome.

### How to build moving average models

An MA model ka order q use karta hai q past disturbances rather than lagged values ka the time series mein a regression-like model. Since hum do not observe the white-noise disturbance values, MA(q) hai not a regression model like the ones hum have seen so far. Rather than use karke least squares, MA(q) models hain estimated use karke maximum likelihood (MLE).

### How to build ARIMA models aur extensions

Autoregressive integrated moving-average ARIMA(p, d, q) models combine AR(p) aur MA(q) processes to leverage the complementarity ka these building blocks aur simplify model development by use karke a more compact form aur reducing the number ka parameters, mein turn reducing the risk ka overfitting.

- statsmodels State-Space Models [docs](https://www.statsmodels.org/dev/statespace.html)

### Code example: forecasting macro fundamentals ke saath ARIMA aur SARIMAX models

hum will build a SARIMAX model ke liye monthly data on an industrial production time series ke liye the 1988-2017 period. See notebook [arima_models](02_arima_models.ipynb) ke liye implementation details.

### How to use time series models to forecast volatility

A particularly important area ka application ke liye univariate time series models hai the prediction ka volatility. The volatility ka financial time series hai usually not constant over time but changes, ke saath bouts ka volatility clustering together. Changes mein variance create challenges ke liye time series forecasting use karke the classical ARIMA models.

### How to build a volatility-forecasting model

The development ka a volatility model ke liye an asset-return series consists ka four steps:
1. Build an ARMA time series model ke liye the financial time series based on the serial dependence revealed by the ACF aur PACF.
2. Test the residuals ka the model ke liye ARCH/GARCH effects, again relying on the ACF aur PACF ke liye the series ka the squared residual.
3. Specify a volatility model if serial correlation effects hain significant, aur jointly estimate the mean aur volatility equations.
4. Check the fitted model carefully aur refine it if necessary.

### Code examples: volatility forecasts

Notebook [arch_garch_models](03_arch_garch_models.ipynb) demonstrate karta hai the usage ka the ARCH library to estimate time series models ke liye volatility forecasting ke saath NASDAQ data.

### Sansadhan (Resources)

- NYU Stern [VLAB](https://vlab.stern.nyu.edu/)
- ARCH Library
    - [docs](https://arch.readthedocs.io/en/latest/index.html) 
    - [examples](http://nbviewer.jupyter.org/github/bashtage/arch/blob/master/examples/univariate_volatility_modeling.ipynb)

## Multivariate Time Series Models

Multivariate time series models hain designed to capture the dynamic ka multiple time series simultaneously aur leverage dependencies across these series ke liye more reliable predictions.

Univariate time-series models like the ARMA approach hain limited to statistical relationships between a target variable aur its lagged values or lagged disturbances aur exogenous series mein the ARMAX case. mein contrast, multivariate time-series models also allow ke liye lagged values ka other time series to affect the target. Yeh effect applies to all series, resulting mein complex interactions.

mein addition to potentially better forecasting, multivariate time series hain also used to gain insights into cross-series dependencies. Udaharan ke liye, mein economics, multivariate time series hain used to understand how policy changes to one variable, such as an interest rate, may affect other variables over different horizons. 

- [New Introduction to Multiple Time Series Analysis, Lütkepohl, Helmut, Springer, 2005](https://www.springer.com/us/book/9783540401728)

### The vector autoregressive (VAR) model

The vector autoregressive VAR(p) model extends the AR(p) model to k series by creating a system ka k equations where each contain karta hai p lagged values ka all k series.

VAR(p) models also require stationarity, so that the initial steps from univariate time-series modeling carry over. First, explore the series aur determine the necessary transformations, aur then apply the augmented Dickey-Fuller test to verify that the stationarity criterion hai met ke liye each series aur apply further transformations otherwise. It can be estimated ke saath OLS conditional on initial information or ke saath MLE, which hai equivalent ke liye normally distributed errors but not otherwise.

If some or all ka the k series hain unit-root non-stationary, they may be cointegrated (see next section). Yeh extension ka the unit root concept to multiple time series means that a linear combination ka two or more series hai stationary aur, hence, mean-reverting. 

### Code example: How to use the VAR model ke liye macro fundamentals forecasts

Notebook [vector_autoregressive_model](04_vector_autoregressive_model.ipynb) demonstrate karta hai how to use `statsmodels` to estimate a VAR model ke liye macro fundamentals time series.

### Sansadhan (Resources)

- `statsmodels` Vector Autoregression [docs](https://www.statsmodels.org/dev/vector_ar.html)
- [Time Series Analysis mein Python ke saath statsmodels](https://conference.scipy.org/proceedings/scipy2011/pdfs/statsmodels.pdf), Wes McKinney, Josef Perktold, Skipper Seabold, SciPY Conference 2011

## Cointegration – time series ke saath a common trend

The concept ka an integrated multivariate series hai complicated by the fact that all the component series ka the process may be individually integrated but the process hai not jointly integrated mein the sense that one or more linear combinations ka the series exist that produce a new stationary series.

mein other words, a combination ka two co-integrated series has a stable mean to which this linear combination reverts. A multivariate series ke saath this characteristic hai said to be co-integrated. Yeh also applies when the individual series hain integrated ka a higher order aur the linear combination reduces the overall order ka integration. 

hum demonstrate two major approaches to testing ke liye cointegration:
- The Engle–Granger two-step method
- The Johansen procedure

### Pairs trading: Statistical arbitrage ke saath cointegration

Statistical arbitrage refers to strategies that employ some statistical model or method to take advantage ka what appears to be relative mispricing ka assets while maintaining a level ka market neutrality.

Pairs trading hai a conceptually straightforward strategy that has been employed by algorithmic traders since at least the mid-eighties (Gatev, Goetzmann, aur Rouwenhorst 2006). The goal hai to find two assets whose prices have historically moved together, track the spread (the difference between their prices), aur, once the spread widens, buy the loser that has dropped below the common trend aur short the winner. If the relationship persists, the long aur/or the short leg will deliver profits as prices converge aur the positions hain closed. 

Yeh approach extends to a multivariate context by forming baskets from multiple securities aur trade one asset against a basket ka two baskets against each other.

mein practice, the strategy requires two steps: 
1. Formation phase: Identify securities that have a long-term mean-reverting relationship. Ideally, the spread should have a high variance to allow ke liye frequent profitable trades while reliably reverting to the common trend.
2. Trading phase: Trigger entry aur exit trading rules as price movements cause the spread to diverge aur converge.

Several approaches to the formation aur trading phases have emerged from increasingly active research mein this area across multiple asset classes over the last several years. The next subsection outlines the key differences before hum dive into an example application.

### Alternative approaches to selecting aur trading comoving assets

A recent comprehensive survey ka pairs trading strategies [Statistical Arbitrage Pairs Trading Strategies: Review
aur Outlook](https://www.iwf.rw.fau.de/files/2016/03/09-2015.pdf), Krauss (2017) identifies four different methodologies plus a number ka other more recent approaches, including ML-based forecasts:

- **Distance** approach: The oldest aur most-studied method identifies candidate pairs ke saath distance metrics like correlation aur use karta hai non-parametric thresholds like Bollinger Bands to trigger entry aur exit trades. The computational simplicity allows ke liye large-scale applications ke saath demonstrated profitability across markets aur asset classes ke liye extended periods ka time since Gatev, et al. (2006). However, performance has decayed more recently.
- **Cointegration** approach: As outlined previously, this approach relies on an econometric model ka a long-term relationship among two or more variables aur allows ke liye statistical tests that promise more reliability than simple distance metrics. Examples mein this category use the Engle-Granger aur Johansen procedures to identify pairs aur baskets ka securities as well as simpler heuristics that aim to capture the concept (Vidyamurthy 2004). Trading rules often resemble the simple thresholds used ke saath distance metrics.
- **Time-series** approach: ke saath a focus on the trading phase, strategies mein this category aim to model the spread as a mean-reverting stochastic process aur optimize entry aur exit rules accordingly (Elliott, Hoek, aur Malcolm 2005). It assumes promising pairs have already been identified.
- **Stochastic control** approach: Similar to the time-series approach, the goal hai to optimize trading rules use karke stochastic control theory to find value aur policy functions to arrive at an optimal portfolio (Liu aur Timmermann 2013). hum will address this type ka approach mein Chapter 21, Reinforcement Learning.
- **Other approaches**: Besides pair identification based on unsupervised learning like principal component analysis (see Chapter 13, Unsupervised Learning) aur statistical models like copulas (Patton 2012), machine learning has become popular more recently to identify pairs based on their relative price or return forecasts (Huck 2019). hum will cover several ML algorithms that can be used ke liye this purpose aur illustrate corresponding multivariate pairs trading strategies mein the coming chapters.

### Code example: Pairs trading mein practice

The **distance approach** identifies pairs use karke the correlation ka (normalized) asset prices or their returns aur hai simple aur orders ka magnitude less computationally intensive than cointegration tests. 
- Notebook [cointegration_tests](05_cointegration_tests.ipynb) illustrates this ke liye a sample ka ~150 stocks ke saath four years ka daily data: it takes ~30ms to compute the correlation ke saath the returns ka an ETF, compared to 18 seconds ke liye a suite ka cointegration tests (use karke statsmodels) - 600x slower.

The speed advantage hai particularly valuable because the number ka potential pairs hai the product ka the number ka candidates to be considered on either side so that evaluating combinations ka 100 stocks aur 100 ETFs requires comparing 10,000 tests (hum’ll discuss the challenge ka multiple testing bias below).

On the other hand, distance metrics do not necessarily select the most profitable pairs: correlation hai maximized ke liye perfect co-movement that mein turn eliminates actual trading opportunities. Empirical studies confirm that the volatility ka the price spread ka cointegrated pairs hai almost twice as high as the volatility ka the price spread ka distance pairs (Huck aur Afawubo 2015).

To balance the tradeoff between computational cost aur the quality ka the resulting pairs, Krauss (2017) recommends a procedure that combines both approaches based on his literature review:
1. Select pairs ke saath a stable spread that shows little drift to reduce the number ka candidates
2. Test the remaining pairs ke saath the highest spread variance ke liye cointegration

Yeh process aims to select cointegrated pairs ke saath lower divergence risk while ensuring more volatile spreads that mein turn generate higher profit opportunities.

A large number ka tests introduce data snooping bias as discussed mein Chapter 6, The Machine Learning Workflow: multiple testing hai likely to increase the number ka false positives that mistakenly reject the null hypothesis ka no cointegration. While statistical significance may not be necessary ke liye profitable trading (Chan 2008), a study ka commodity pairs (Cummins aur Bucca 2012) shows that controlling the familywise error rate to improve the tests’ power according to Romano aur Wolf (2010) can lead to better performance.

#### Computing distance-based heuristics to identify cointegrated pairs

- Notebook [cointegration_tests](05_cointegration_tests.ipynb) takes a closer look at how predictive various heuristics ke liye the degree ka comovement ka asset prices hain ke liye the result ka cointegration tests. The example code use karta hai a sample ka 172 stocks aur 138 ETFs traded on the NYSE aur NASDAQ ke saath daily data from 2010 - 2019 provided by Stooq. 

The securities represent the largest average dollar volume over the sample period mein their respective class; highly correlated aur stationary assets have been removed. See the notebook [create_datasets](../data/create_datasets.ipynb) mein the data folder ka the GitHub repository ke liye downloading ke liye instructions on how to obtain the data aur the notebook cointegration_tests ke liye the relevant code aur additional preprocessing aur exploratory details.

#### Precomputing the cointegration tests

Notebook [statistical_arbitrage_with_cointegrated_pairs](06_statistical_arbitrage_with_cointegrated_pairs.ipynb) implements a statistical arbitrage strategy based on cointegration ke liye the sample ka stocks aur ETFs aur the 2017-2019 period.

It first generates aur stores the cointegration tests ke liye all candidate pairs aur the resulting trading signals before hum backtest a strategy based on these signals given the computational intensity ka the process.

### Sansadhan (Resources)

- Quantopian offers various resources on pairs trading:
    - [Introduction to Pairs Trading](https://www.quantopian.com/lectures/introduction-to-pairs-trading)
    - [Quantopian Johansen](https://www.quantopian.com/posts/trading-baskets-co-integrated-with-spy)
    - [Quantopian PT](https://www.quantopian.com/posts/how-to-build-a-pairs-trading-strategy-on-quantopian)
    - [Pairs Trading Basics: Correlation, Cointegration And Strategy](https://blog.quantinsti.com/pairs-trading-basics/)
- Additional blog posts include:
    - [Pairs Trading using Data-Driven Techniques: Simple Trading Strategies Part 3](https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a)
    - [Pairs Trading Johansen & Kalman](https://letianzj.github.io/kalman-filter-pairs-trading.html)
    - [Copulas](https://twiecki.io/blog/2018/05/03/copulas/) by Thomas Wiecki
