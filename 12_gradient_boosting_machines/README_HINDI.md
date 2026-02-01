# Apni Trading Strategy ko Boost Karein

Yeh chapter explore karta hai boosting, another ensemble learning algorithm typically based on decision trees that often produces even better results the [Random Forests](../10_decision_trees_random_forests). 

The key difference hai that boosting, mein its original AdaBoost version, modifies the training data ke liye each tree based on the cumulative errors made by the model before adding the new tree. Random Forests, mein contrast, use bagging to train many trees independently from each other use karke randomized versions ka the training set. While Random Forests can be trained mein parallel, boosting proceeds sequentially use karke reweighted versions ka the data. State-ka-the-art boosting implementations also adopt the randomization strategies ka random forests.

hai chapter mein, hum dekhenge how boosting has evolved into one ka the most successful machine learning algorithms over the last three decades. At the time ka writing, it has come to dominate machine learning competitions ke liye structured data (as opposed to high-dimensional images or speech, ke liye example, where the relationship between the input aur output hai more complex, aur deep learning excels at). More specifically, mein this chapter hum will cover the following topics:

## Vishay-suchi (Content)

1. [Getting started: adaptive boosting](#getting-started-adaptive-boosting)
    * [The AdaBoost algorithm](#the-adaboost-algorithm)
    * [Code Example: AdaBoost with sklearn](#code-example-adaboost-with-sklearn)
2. [Gradient boosting - ensembles ke liye most tasks ](#gradient-boosting---ensembles-ke liye-most-tasks-)
    * [How to train and tune GBM models](#how-to-train-and-tune-gbm-models)
    * [Code Example: Gradient boosting with scikit-learn](#code-example-gradient-boosting-with-scikit-learn)
3. [use karke XGBoost, LightGBM aur CatBoost](#use karke-xgboost-lightgbm-aur-catboost)
4. [Code Example: A long-short trading strategy ke saath gradient boosting](#code-example-a-long-short-trading-strategy-ke saath-gradient-boosting)
    * [Preparing the data](#preparing-the-data)
    * [How to generate signals with LightGBM and CatBoost models](#how-to-generate-signals-with-lightgbm-and-catboost-models)
    * [Evaluating the trading signals](#evaluating-the-trading-signals)
    * [Creating out-of-sample predictions](#creating-out-of-sample-predictions)
    * [Defining and backtesting the long-short strategy](#defining-and-backtesting-the-long-short-strategy)
5. [A peek into the black box: How to interpret GBM results](#a-peek-into-the-black-box-how-to-interpret-gbm-results)
    * [Code example: attributing feature importance with LightGBM](#code-example-attributing-feature-importance-with-lightgbm)
        - [Feature importance](#feature-importance)
        - [Partial dependence plots](#partial-dependence-plots)
        - [SHapley Additive exPlanations (SHAP Values)](#shapley-additive-explanations-shap-values)
6. [An intraday strategy ke saath Algoseek aur LightGBM](#an-intraday-strategy-ke saath-algoseek-aur-lightgbm)
    * [Code example: engineering intraday features](#code-example-engineering-intraday-features)
    * [Code example: tuning a LightGBM model and evaluating the forecasts](#code-example-tuning-a-lightgbm-model-and-evaluating-the-forecasts)
7. [Resources](#resources)
    * [XGBoost](#xgboost)
    * [LightGBM](#lightgbm)
    * [CatBoost](#catboost)


## Shuruat: Adaptive Boosting

Like bagging, boosting combines base learners into an ensemble. Boosting was initially developed ke liye classification problems, but can also be used ke liye regression, aur has been called one ka the most potent learning ideas introduced mein the last 20 years (as described mein [Elements ka Statistical Learning](http://web.stanford.edu/~hastie/ElemStatLearn/) by Trevor Hastie, et al). Like bagging, it hai a general method or metamethod that can be applied to many statistical learning models.

Niche diye gaye sections briefly introduce AdaBoost aur then focus on the gradient boosting model, as well as several state-ka-the-art implementations ka this algorithm. 

### AdaBoost Algorithm

AdaBoost hai a significant departure from bagging, which builds ensembles on very deep trees to reduce bias. AdaBoost, mein contrast, grows shallow trees as weak learners, often producing superior accuracy ke saath stumps—that hai, trees formed by a single split. The algorithm starts ke saath an equal-weighted training set aur then successively alters the sample distribution. After each iteration, AdaBoost increases the weights ka incorrectly classified observations aur reduces the weights ka correctly predicted samples so that subsequent weak learners focus more on particularly difficult cases. Once trained, the new decision tree hai incorporated into the ensemble ke saath a weight that reflects its contribution to reducing the training error.

- [A Decision-Theoretic Generalization ka On-Line Learning aur an Application to Boosting](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf), Y. Freund, R. Schapire, 1997.

### Code Example: AdaBoost ke saath sklearn

Ke hisse ke roop mein its ensemble module, sklearn provide karta hai an [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) implementation that supports two or more classes. 

The code examples ke liye this section hain mein the notebook [gbm_baseline](01_gbm_baseline.ipynb) that compares the performance ka various algorithms ke saath a dummy classifier that always predicts the most frequent class.

The algorithms mein this chapter use a dataset generated by the notebook [feature-engineering](../04_alpha_factor_research/00_data/feature_engineering.ipynb) from [Chapter 4 on Alpha Factor Research](../04_alpha_factor_research) that needs to be executed first.

- `sklearn` AdaBoost [docs](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

## Gradient Boosting - Adhiktar tasks ke liye ensembles 

The main idea behind the resulting Gradient Boosting Machines (GBM) algorithm hai the training ka the base learners to learn the negative gradient ka the current loss function ka the ensemble. As a result, each addition to the ensemble directly contributes to reducing the overall training error given the errors made by prior ensemble members. Since each new member represents a new function ka the data, gradient boosting hai also said to optimize over the functions hm mein an additive fashion. 

- [Greedy function approximation: A gradient boosting machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf), Jerome H. Friedman, 1999

### GBM Models ko kaise train aur tune karein

The two key drivers ka gradient boosting performance hain the size ka the ensemble aur the complexity ka its constituent decision trees. The control ka complexity ke liye decision trees aims to avoid learning highly specific rules that typically imply a very small number ka samples mein leaf nodes. hum covered the most effective constraints used to limit the ability ka a decision tree to overfit to the training data mein [Chapter 4 on Decision Trees aur Random Forests](../10_decision_trees_random_forests).

mein addition to directly controlling the size ka the ensemble, there hain various regularization techniques, such as shrinkage, that hum encountered mein the context ka the Ridge aur Lasso linear regression models mein [Chapter 7, Linear Models – Regression aur Classification](../07_linear_models). Furthermore, the randomization techniques used mein the context ka random forests hain also commonly applied to gradient boosting machines.

### Code Example: Gradient boosting ke saath scikit-learn

The ensemble module ka sklearn contain karta hai an implementation ka gradient boosting trees ke liye regression aur classification, both binary aur multiclass.

Notebook [boosting_baseline](./01_boosting_baseline.ipynb) demonstrate karta hai how to run cross-validation ke liye the [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

Notebook [sklearn_gbm_tuning](02_sklearn_gbm_tuning.ipynb) shows how to [GridSearchCV]() to search ke liye the best set ka parameters. Yeh can be very time-consuming to run. 

Notebook [sklearn_gbm_tuning_results](03_sklearn_gbm_tuning_results.ipynb) displays some ka the results that can be obtained.

- `scikit-klearn` Gradient Boosting [docs](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

## XGBoost, LightGBM aur CatBoost ka upyog

Pichle kuch saalon mein, several new gradient boosting implementations have used various innovations that accelerate training, improve resource efficiency, aur allow the algorithm to scale to very large datasets. The new implementations aur their sources hain as follows:
- [XGBoost](https://github.com/dmlc/xgboost) (extreme gradient boosting), started mein 2014 by Tianqi Chen at the University ka Washington 
- [LightGBM](https://github.com/Microsoft/LightGBM), first released mein January 2017, by Microsoft
- [CatBoost](https://tech.yandex.com/catboost/), first released mein April 2017 by Yandex

The book reviews the numerous algorithmic innovations that have emerged over time aur subsequently converged (so that most features hain available ke liye all implementations) before illustrating their implementation.



## Code Example: Gradient boosting ke saath Long-Short Trading Strategy

mein this section, hum’ll design, implement, aur evaluate a trading strategy ke liye US equities driven by daily return forecasts produced by gradient boosting models. 

As mein the previous examples, hum’ll lay out a framework aur build a specific example that you can adapt to run your own experiments. There hain numerous aspects that you can vary, from the asset class aur investment universe to more granular aspects like the features, holding period, or trading rules. See, ke liye example, the Alpha Factor Library mein the Appendix ke liye numerous additional features.

### Preparing the data

hum use the Quandl Wiki data to engineer a few simple features (see notebook [preparing_the_model_data](04_preparing_the_model_data.ipynb) ke liye details) aur select a model use karke 2015/16 as validation period aur run an out-ka-sample test ke liye 2017.

### How to generate signals ke saath LightGBM aur CatBoost models

hum’ll keep the trading strategy simple aur only use a single machine learning (ML) signal; a real-life application will likely use multiple signals from different sources, such as complementary ML models trained on different datasets or ke saath different lookahead or lookback periods. It would also use sophisticated risk management from simple stop-loss to value-at-risk analysis.

XGBoost, LightGBM, aur CatBoost offer interfaces ke liye multiple languages, including Python, aur have both a sklearn interface that hai compatible ke saath other sklearn features, such as GridSearchCV aur their own methods to train aur predict gradient boosting models. Notebook [gbm_baseline](01_gbm_baseline.ipynb) illustrates the use ka the sklearn interface ke liye each implementation. The library methods hain often better documented aur hain also easy to use, so hum'll use them to illustrate the use ka these models.

The process entails the creation ka library-specific data formats, the tuning ka various hyperparameters, aur the evaluation ka results that hum will describe mein the following sections. 

- Notebook [trading_signals_with_lightgbm_and_catboost](05_trading_signals_with_lightgbm_and_catboost.ipynb) cross-validates a range ka hyperparameter options to optimize the models' predictive performance.

### Evaluating the trading signals

Notebook [evaluate_trading_signals](06_evaluate_trading_signals.ipynb) demonstrate karta hai how to evaluate the trading signals.

### Creating out-ka-sample predictions

Notebook [making_out_of_sample_predictions](08_making_out_of_sample_predictions.ipynb) shows how to create predictions ke liye the best-performing models.

### Defining aur backtesting the long-short strategy

Notebook [backtesting_with_zipline](09_backtesting_with_zipline.ipynb) create karta hai a strategy based on the model predictions, simulates its historical performance use karke [Zipline](https://zipline.ml4trading.io/, aur evaluates the result use karke [Pyfolio](https://github.com/quantopian/pyfolio. 

## Black box mein ek jhalak: GBM results ko kaise interpret karein

Yeh samajhna ki kyun a model predicts a certain outcome hai very important ke liye several reasons, including trust, actionability, accountability, aur debugging. Insights into the nonlinear relationship between features aur the outcome uncovered by the model, as well as interactions among features, hain also ka value when the goal hai to learn more about the underlying drivers ka the phenomenon under study.

### Code example: attributing feature importance ke saath LightGBM

Notebook [model_interpretation](06_model_interpretation.ipynb) illustrates the following methods.

#### Feature importance

There hain three primary ways to compute global feature importance values:
- Gain: This classic approach introduced by Leo Breiman mein 1984 use karta hai the total reduction ka loss or impurity contributed by all splits ke liye a given feature. The motivation hai largely heuristic, but it hai a commonly used method to select features.
- Split count: This hai an alternative approach that counts how often a feature hai used to make a split decision, based on the selection ka features ke liye this purpose based on the resultant information gain.
- Permutation: This approach randomly permutes the feature values mein a test set aur measures how much the model's error changes, assuming that an important feature should create a large increase mein the prediction error. Different permutation choices lead to alternative implementations ka this basic approach.

#### Partial dependence plots

mein addition to the summary contribution ka individual features to the model's prediction, partial dependence plots visualize the relationship between the target variable aur a set ka features. The nonlinear nature ka gradient boosting trees causes this relationship to depend on the values ka all other features.

#### SHapley Additive exPlanations (SHAP Values)

At the 2017 NIPS conference, Scott Lundberg aur Su-mein Lee from the University ka Washington presented a new aur more accurate approach to explaining the contribution ka individual features to the output ka tree ensemble models called [SHapley Additive exPlanations](https://github.com/slundberg/shap), or SHAP values.

Yeh new algorithm departs from the observation that feature-attribution methods ke liye tree ensembles, such as the ones hum looked at earlier, hain inconsistent — that hai, a change mein a model that increases the impact ka a feature on the output can lower the importance values ke liye this feature.

SHAP values unify ideas from collaborative game theory aur local explanations, aur have been shown to be theoretically optimal, consistent, aur locally accurate based on expectations. Most importantly, Lundberg aur Lee have developed an algorithm that manages to reduce the complexity ka computing these model-agnostic, additive feature-attribution methods from O(TLDM) to O(TLD2), where T aur M hain the number ka trees aur features, respectively, aur D aur L hain the maximum depth aur number ka leaves across the trees. 

Yeh important innovation permits the explanation ka predictions from previously intractable models ke saath thousands ka trees aur features mein a fraction ka a second. An open source implementation became available mein late 2017 aur hai compatible ke saath XGBoost, LightGBM, CatBoost, aur sklearn tree models. 

Shapley values originated mein game theory as a technique ke liye assigning a value to each player mein a collaborative game that reflects their contribution to the team's success. SHAP values hain an adaptation ka the game theory concept to tree-based models aur hain calculated ke liye each feature aur each sample. They measure how a feature contributes to the model output ke liye a given observation. ke liye this reason, SHAP values provide differentiated insights into how the impact ka a feature varies across samples, which hai important given the role ka interaction effects mein these nonlinear models.

SHAP values provide granular feature attribution at the level ka each individual prediction, aur enable much richer inspection ka complex models through (interactive) visualization. The SHAP summary scatterplot displayed at the beginning ka this section offers much more differentiated insights than a global feature-importance bar chart. Force plots ka individual clustered predictions allow ke liye more detailed analysis, while SHAP dependence plots capture interaction effects aur, as a result, provide more accurate aur detailed results than partial dependence plots.

## Algoseek aur LightGBM ke saath ek Intraday Strategy

Yeh section aur the notebooks will be updated once Algoseek makes the sample data available.

### Code example: engineering intraday features

Notebook [intraday_features](10_intraday_features.ipynb) create karta hai features from minute-bar trade aur quote data.

### Code example: tuning a LightGBM model aur evaluating the forecasts

Notebook [intraday_model](11_intraday_model.ipynb) optimizes a LightGBM boosting model, generates out-ka-sample predictions, aur evaluates the result.

## Sansadhan (Resources)

- [xgboost - LightGBM Parameter Comparison](https://sites.google.com/view/lauraepp/parameters)
- [xgboost vs LightGBM Benchmarks](https://sites.google.com/view/lauraepp/new-benchmarks)
- [Depth- vs Leaf-wise growth](https://datascience.stackexchange.com/questions/26699/decision-trees-leaf-wise-best-first-aur-level-wise-tree-traverse)
- [Rashmi Korlakai Vinayak, Ran Gilad-Bachrach. “DART: Dropouts meet Multiple Additive Regression Trees.”](http://www.jmlr.org/proceedings/papers/v38/korlakaivinayak15.pdf)

### XGBoost

- [GitHub Repo](https://github.com/dmlc/xgboost)
- [Documentation](https://xgboost.readthedocs.io)
- [Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Accelerating the XGBoost algorithm use karke GPU computing. Mitchell R, Frank E., 2017, PeerJ Computer Science 3:e127](https://peerj.com/articles/cs-127/)
- [XGBoost: Scalable GPU Accelerated Learning, Rory Mitchell, Andrey Adinets, Thejaswi Rao, 2018](http://arxiv.org/abs/1806.11248)
- [Nvidia Parallel Forall: Gradient Boosting, Decision Trees aur XGBoost ke saath CUDA, Rory Mitchell, 2017](https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/)
- [Awesome XGBoost](https://github.com/dmlc/xgboost/tree/master/demo)

### LightGBM

- [GitHub Repo](https://github.com/Microsoft/LightGBM)
- [Documentation](https://lightgbm.readthedocs.io/en/latest/index.html)
- [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [On Grouping ke liye Maximum Homogeneity](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.1958.10501479#.W8_3pXX24UE)

### CatBoost

- [Python API](https://tech.yandex.com/catboost/doc/dg/concepts/python-quickstart-docpage/)
- [CatBoost: unbiased boosting ke saath categorical features](https://arxiv.org/abs/1706.09516)
- [CatBoost: gradient boosting ke saath categorical features](http://learningsys.org/nips17/assets/papers/paper_11.pdf)