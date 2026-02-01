# Bayesian ML: From recession forecasts to dynamic pairs trading

mein this chapter, hum will introduce Bayesian approaches to machine learning (ML) aur how their different perspective on uncertainty adds value when developing aur evaluating trading strategies.

Bayesian statistics allows us to quantify uncertainty about future events aur refine our estimates mein a principled way as new information arrives. Yeh dynamic approach adapts well to the evolving nature ka financial markets. It hai particularly useful when there hain fewer relevant data aur hum require methods that systematically integrate prior knowledge or assumptions.

hum will see that Bayesian approaches to machine learning allow ke liye richer insights into the uncertainty around statistical metrics, parameter estimates, aur predictions. The applications range from more granular risk management to dynamic updates ka predictive models that incorporate changes mein the market environment. The Black-Litterman approach to asset allocation (see [Chapter 5, Portfolio Optimization aur Performance Evaluation](../05_strategy_evaluation) can be interpreted as a Bayesian model. It computes the expected return ka an asset as an average ka the market equilibrium aur the investor’s views, weighted by each asset’s volatility, cross-asset correlations, aur the confidence mein each forecast.

## Vishay-suchi (Content)

1. [How Bayesian Machine Learning Works](#how-bayesian-machine-learning-works)
        - [References](#references)
    * [How to update assumptions from empirical evidence](#how-to-update-assumptions-from-empirical-evidence)
    * [Exact Inference: Maximum a Posterior Estimation](#exact-inference-maximum-a-posterior-estimation)
        - [How to keep inference simple: Conjugate Priors](#how-to-keep-inference-simple-conjugate-priors)
        - [Code example: How to dynamically estimate the probabilities of asset price moves](#code-example-how-to-dynamically-estimate-the-probabilities-of-asset-price-moves)
    * [Deterministic and stochastic approximate inference](#deterministic-and-stochastic-approximate-inference)
2. [Probabilistic Programming ke saath PyMC3](#probabilistic-programming-ke saath-pymc3)
    * [Bayesian ML with Theano](#bayesian-ml-with-theano)
    * [The PyMC3 workflow](#the-pymc3-workflow)
    * [Code example: Predicting a recession with PyMC3](#code-example-predicting-a-recession-with-pymc3)
    * [The Data: Leading Recession Indicators](#the-data-leading-recession-indicators)
        - [Model Definition: Bayesian Logistic Regression](#model-definition-bayesian-logistic-regression)
3. [Bayesian ML ke liye Trading](#bayesian-ml-ke liye-trading)
    * [Code Example: Bayesian Sharpe ratio for performance comparison](#code-example-bayesian-sharpe-ratio-for-performance-comparison)
    * [Code Example: Bayesian Rolling Regression for Pairs Trading](#code-example-bayesian-rolling-regression-for-pairs-trading)
    * [Code Example: Stochastic Volatility Models](#code-example-stochastic-volatility-models)
4. [Resources](#resources)
    * [PyMC3](#pymc3)
    * [Alternative probabilistic programming libraries](#alternative-probabilistic-programming-libraries)


## How Bayesian Machine Learning Works

Classical statistics hai said to follow the frequentist approach because it interprets probability as the relative frequency ka an event over the long run, i.e. after observing a large number ka trials. mein the context ka probabilities, an event hai a combination ka one or more elementary outcomes ka an experiment, such as any ka six equal results mein rolls ka two dice or an asset price dropping by 10 percent or more on a given day). 

Bayesian statistics, mein contrast, views probability as a measure ka the confidence or belief mein the occurrence ka an event. The Bayesian perspective, thus, leaves more room ke liye subjective views aur differences mein opinions than the frequentist interpretation. Yeh difference hai most striking ke liye events that do not happen often enough to arrive at an objective measure ka long-term frequency.

Put differently, frequentist statistics assumes that data hai a random sample from a population aur aims to identify the fixed parameters that generated the data. Bayesian statistics, mein turn, takes the data as given aur considers the parameters to be random variables ke saath a distribution that can be inferred from data. As a result, frequentist approaches require at least as many data points as there hain parameters to be estimated. Bayesian approaches, on the other hand, hain compatible ke saath smaller datasets, aur well suited ke liye online learning from one sample at a time.

The Bayesian view hai very useful ke liye many real-world events that hain rare or unique, at least mein important respects. Examples include the outcome ka the next election or the question ka whether the markets will crash within three months. mein each case, there hai both relevant historical data as well as unique circumstances that unfold as the event approaches.

hum first introduce the Bayes theorem that crystallizes the concept ka updating beliefs by combining prior assumptions ke saath new empirical evidence aur compare the resulting parameter estimates ke saath their frequentist counterparts. hum then demonstrate two approaches to Bayesian statistical inference, namely conjugate priors aur approximate inference that produce insights into the posterior distribution ka latent, i.e. unobserved parameters, such as the expected value:

1. Conjugate priors facilitate the updating process by providing a closed-form solution that allows us to precisely compute the solution. However, such exact, analytical methods hain not always available.
2. Approximate inference simulates the distribution that results from combining assumptions aur data aur use karta hai samples from this distribution to compute statistical insights.

#### References

- [Bayesian Methods ke liye Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-aur-Bayesian-Methods-ke liye-Hackers/)
- [Andrew Gelman's Blog](https://andrewgelman.com/)
- [Thomas Wiecki's Blog](https://twiecki.github.io/)

### How to update assumptions from empirical evidence

The theorem that Reverend Thomas Bayes came up ke saath over 250 years ago use karta hai fundamental probability theory to prescribe how probabilities or beliefs should change as relevant new information arrives as captured by John Maynard Keynes’ quote “When the facts change, I change my mind. What do you do, sir?”.

- [Bayes' rule: Guide](https://arbital.com/p/bayes_rule/?l=1zq)
- [Bayesian Updating ke saath Continuous Priors](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-aur-statistics-spring-2014/readings/MIT18_05S14_Reading13a.pdf), MIT Open Courseware, 18.05 Introduction to Probability aur Statistics

### Exact Inference: Maximum a Posterior Estimation

Practical applications ka Bayes’ rule to exactly compute posterior probabilities hain quite limited because the computation ka the evidence term mein the denominator hai quite challenging.

#### How to keep inference simple: Conjugate Priors
A prior distribution hai conjugate ke saath respect to the likelihood when the resulting posterior hai ka the same type ka distribution as the prior except ke liye different parameters. The conjugacy ka prior aur likelihood implies a closed-form solution ke liye the posterior that facilitates the update process aur avoids the need to use numerical methods to approximate the posterior.

#### Code example: How to dynamically estimate the probabilities ka asset price moves

Notebook [updating_conjugate_priors](01_updating_conjugate_priors.ipynb) demonstrate karta hai how to use a conjugate prior to update price movement estimates from S&P 500 samples.

### Deterministic aur stochastic approximate inference

ke liye most models ka practical relevance, it will not be possible to derive the exact posterior distribution analytically aur compute expected values ke liye the latent parameters.

Although ke liye some applications the posterior distribution over unobserved parameters
will be ka interest, most often it hai primarily required to evaluate expectations, e.g. to make predictions. mein such situations, hum can rely on approximate inference:
- **Stochastic techniques** based on Markov Chain Monte Carlo (MCMC) sampling have popularized the use ka Bayesian methods across many domains. They generally have the property to converge to the exact result. mein practice, sampling methods can be computationally demanding aur hain often limited to small-scale problems. 
    - [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf), Michael Betancourt, 2018
    - [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://arxiv.org/abs/1111.4246), Matthew D. Hoffman, Andrew Gelman, 2011
    - [ML, MAP, and Bayesian — The Holy Trinity of Parameter Estimation and Data Prediction](https://engineering.purdue.edu/kak/Trinity.pdf)

- **Deterministic methods** called variational inference or variational Bayes hain based on analytical approximations to the posterior distribution aur can scale well to large applications. They make simplifying assumptions, e.g., that the posterior factorizes mein a particular way or it has a specific parametric form such as a Gaussian. Hence, they do not generate exact results aur can be used as complements to sampling methods.
    - [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf), David Blei et al, 2018

## Probabilistic Programming ke saath PyMC3

Probabilistic programming provide karta hai a language to describe aur fit probability distributions so that hum can design, encode aur automatically estimate aur evaluate complex models. It aims to abstract away some ka the computational aur analytical complexity to allow us to focus on the conceptually more straightforward aur intuitive aspects ka Bayesian reasoning aur inference.
The field has become quite dynamic since new languages emerged since Uber open-sourced Pyro (based on PyTorch) aur Google more recently added a probability module to TensorFlow. 

### Bayesian ML ke saath Theano

- [PyMC3](https://docs.pymc.io/) was released mein January 2017 to add Hamiltonian MC methods to the Metropolis-Hastings sampler used mein PyMC2 (released 2012). PyMC3 use karta hai [Theano](http://www.deeplearning.net/software/theano/) as its computational backend ke liye dynamic C compilation aur automatic differentiation. Theano hai a matrix-focused aur GPU-enabled optimization library developed at Yoshua Bengio’s Montreal Institute ke liye Learning Algorithms (MILA) that inspired TensorFlow. MILA recently ceased to further develop Theano due to the success ka newer deep learning libraries (see chapter 16 ke liye details). 
- [PyMC4](https://github.com/pymc-devs/pymc4), planned ke liye 2019, will use TensorFlow instead, ke saath presumably limited impact on the API.

### The PyMC3 workflow

PyMC3 aims ke liye intuitive aur readable, yet powerful syntax that reflects how statisticians describe models. The modeling process generally follows these three steps:
1) Encode a probability model by defining: 
    1) The prior distributions that quantify knowledge and uncertainty about latent variables
    2) The likelihood function that conditions the parameters on observed data
2) Analyze the posterior use karke one ka the options described mein the previous section:
    1) Obtain a point estimate using MAP inference
    2) Sample from the posterior using MCMC methods
    3)Approximate the posterior using variational Bayes
3) Check your model use karke various diagnostic tools
4) Generate predictions

- [Documentation](https://docs.pymc.io/)
- [Probabilistic Programming mein Python use karke PyMC](https://arxiv.org/abs/1507.08050), Salvatier et al 2015
- [Theano: A Python framework ke liye fast computation ka mathematical expressions](https://pdfs.semanticscholar.org/6b57/0069f14c7588e066f7138e1f21af59d62e61.pdf), Al-Rfou et al, 2016
- [Bayesian Methods ke liye Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-aur-Bayesian-Methods-ke liye-Hackers)
- [Bad Traces, or, Don't Use Metropolis](https://colindcarroll.com/2018/01/01/bad-traces-or-dont-use-metropolis/)
- PyMC 4 on [GitHub](https://github.com/pymc-devs/pymc4) ke saath design guide aur a usage examples.

### Code example: Predicting a recession ke saath PyMC3

Notebook [pymc3_workflow](02_pymc3_workflow.ipynb) illustrates various aspects ka the PyMC3 workflow use karke a simple logistic regression to model the prediction ka a recession.

### The Data: Leading Recession Indicators

hum will use a small aur simple dataset so hum can focus on the workflow. hum use the Federal Reserve’s Economic Data (FRED) service (see Chapter 2) to download the US recession dates as defined by the National Bureau ka Economic Research. hum also source four variables that hain commonly used to predict the onset ka a recession (Kelley 2019) aur available via FRED, namely:
- The long-term spread ka the treasury yield curve, defined as the difference between the ten-year aur the three-month Treasury yield.
- The University ka Michigan’s consumer sentiment indicator
- The National Financial Conditions Index (NFCI), aur
- The NFCI nonfinancial leverage subindex.

#### Model Definition: Bayesian Logistic Regression

As discussed mein [Linear Models](../07_linear_models), logistic regression estimates a linear relationship between a set ka features aur a binary outcome, mediated by a sigmoid function to ensure the model produces probabilities. The frequentist approach resulted mein point estimates ke liye the parameters that measure the influence ka each feature on the probability that a data point belongs to the positive class, ke saath confidence intervals based on assumptions about the parameter distribution. 

Bayesian logistic regression, mein contrast, estimates the posterior distribution over the parameters itself. The posterior allows ke liye more robust estimates ka what hai called a Bayesian credible interval ke liye each parameter ke saath the benefit ka more transparency about the model’s uncertainty.

Notebook [pymc3_workflow](02_pymc3_workflow.ipynb) demonstrate karta hai the PYMC3 workflow, including:
- MAP Inference
- Markov Chain Monte Carlo Estimate 
    - Metropolis-Hastings
    - NUTS Sampler
- Variational Inference
- Model Diagnostics
    - Energy and Forest Plots
    - Posterior Predictive Checks (PPD), and 
    - Credible Intervals (CI)
- Prediction
- MCMC Sampler Animation
    
## Bayesian ML ke liye Trading

Now that hum hain familiar ke saath the Bayesian approach to ML aur probabilistic programming ke saath PyMC3, let’s explore a few relevant trading-related applications, namely 
- modeling the Sharpe ratio as a probabilistic model ke liye more insightful performance comparison
- computing pairs trading hedge ratios use karke Bayesian linear regression
- analyzing linear time series models from a Bayesian perspective

### Code Example: Bayesian Sharpe ratio ke liye performance comparison

Notebook [bayesian_sharpe_ratio](03_bayesian_sharpe_ratio.ipynb) illustrates how to define the Sharpe ratio (SR) as a probabilistic model use karke PyMC3, aur how to compare its posterior distributions ke liye different return series. 

The Bayesian estimation ke liye two series offers very rich insights because it provide karta hai the complete distributions ka the credible values ke liye the effect size, the group SR means aur their difference, as well as standard deviations aur their difference. The Python implementation hai due to Thomas Wiecki aur was inspired by the R package BEST (Meredith aur Kruschke 2018), see 'Resources' below.

Relevant use cases ka a Bayesian SR include the analysis ka differences between alternative strategies, or between a strategy’s mein-sample return relative to its out-ka-sample return (see the notebook bayesian_sharpe_ratio ke liye details). The Bayesian Sharpe ratio hai also part ka pyfolio’s Bayesian tearsheet.

### Code Example: Bayesian Rolling Regression ke liye Pairs Trading

The last [chapter](../09_time_series_models) introduced pairs trading as a popular trading strategy that relies on the **cointegration** ka two or more assets. Given such assets, hum need to estimate the hedging ratio to decide on the relative magnitude ka long aur short positions. A basic approach use karta hai linear regression. 

Notebook [rolling_regression](04_rolling_regression.ipynb) illustrates how Bayesian linear regression tracks changes mein the relationship between two assets over time. It follows Thomas Wiecki’s example (see 'Resources' below).

### Code Example: Stochastic Volatility Models

As discussed mein the chapter [Time Series Models](../09_time_series_models), asset prices have time-varying volatility. mein some periods, returns hain highly variable, while mein others very stable. 

Stochastic volatility models model this ke saath a latent volatility variable, modeled as a stochastic process. The  No-U-Turn Sampler was introduced use karke such a model, aur the notebook [stochastic_volatility](05_stochastic_volatility.ipynb) illustrates this use case.

## Sansadhan (Resources)

### PyMC3

Thomas Wiecki, one ka the main PyMC3 authors who also leads Data Science at Quantopian has created several examples that the following sections follow aur build on. The PyMC3 documentation has many additional tutorials.

- PyMC3 [Tutorials](https://docs.pymc.io/nb_tutorials/index.html)
- [Tackling the Poor Assumptions ka Naive Bayes Text Classifiers](http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf), Rennie, et al, MIT SAIL, 2003
- [On Discriminative vs Generative Classifiers: A comparison ka logistic regression aur naive Bayes](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf), Jordan, Ng, 2002
- [Bayesian estimation supersedes the t test](http://www.indiana.edu/~kruschke/BEST/BEST.pdf), John K. Kruschke, Journal ka Experimental Psychology, 2012
- [Automatic Differentiation Variational Inference](https://arxiv.org/pdf/1603.00788.pdf)

### Alternative probabilistic programming libraries
- [Probabilistic Programming](http://www.probabilistic-programming.org/wiki/Home) Community Repository ke saath links to papers aur software
- [Stan](https://mc-stan.org/)
- [Edward](http://edwardlib.org/)
- [TensorFlow Probability](https://github.com/tensorflow/probability)
- [Pyro](http://pyro.ai/)

