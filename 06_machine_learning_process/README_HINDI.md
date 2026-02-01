# The Machine Learning Workflow

Yeh chapter starts part 2 ka this book where hum illustrate how you can use a range ka supervised aur unsupervised machine learning (ML) models ke liye trading. hum will explain each model's assumptions aur use cases before hum demonstrate relevant applications use karke various Python libraries. The categories ka models that hum will cover mein parts 2-4 include:

- Linear models ke liye the regression aur classification ka cross-section, time series, aur panel data
- Generalized additive models, including nonlinear tree-based models, such as decision trees
- Ensemble models, including random forest aur gradient-boosting machines
- Unsupervised linear aur nonlinear methods ke liye dimensionality reduction aur clustering
- Neural network models, including recurrent aur convolutional architectures
- Reinforcement learning models

hum will apply these models to the market, fundamental, aur alternative data sources introduced mein the first part ka this book. hum will build on the material covered so far by demonstrating how to embed these models mein a trading strategy that translates model signals into trades, how to optimize portfolio, aur how to evaluate strategy performance.

There hain several aspects that many ka these models aur their applications have mein common. Yeh chapter covers these common aspects so that hum can focus on model-specific usage mein the following chapters. They include the overarching goal ka learning a functional relationship from data by optimizing an objective or loss function. They also include the closely related methods ka measuring model performance.

hum distinguish between unsupervised aur supervised learning aur outline use cases ke liye algorithmic trading. hum contrast supervised regression aur classification problems, the use ka supervised learning ke liye statistical inference ka relationships between input aur output data ke saath its use ke liye the prediction ka future outputs. hum also illustrate how prediction errors hain due to the model's bias or variance, or because ka a high noise-to-signal ratio mein the data. Most importantly, hum present methods to diagnose sources ka errors like overfitting aur improve your model's performance.

If you hain already quite familiar ke saath ML, feel free to skip ahead aur dive right into learning how to use ML models to produce aur combine alpha factors ke liye an algorithmic trading strategy.

## Vishay-suchi (Content)

1. [How machine learning from data works](#how-machine-learning-from-data-works)
    * [The key challenge: Finding the right algorithm for the given task](#the-key-challenge-finding-the-right-algorithm-for-the-given-task)
    * [Supervised Learning: teaching a task by example](#supervised-learning-teaching-a-task-by-example)
    * [Unsupervised learning: Exploring data to identify useful patterns](#unsupervised-learning-exploring-data-to-identify-useful-patterns)
        - [Use cases for trading strategies: From risk management to text processing](#use-cases-for-trading-strategies-from-risk-management-to-text-processing)
    * [Reinforcement learning: Learning by doing, one step at a time](#reinforcement-learning-learning-by-doing-one-step-at-a-time)
2. [The Machine Learning Workflow](#the-machine-learning-workflow)
    * [Code Example: ML workflow with K-nearest neighbors](#code-example-ml-workflow-with-k-nearest-neighbors)
3. [Frame the problem: goals & metrics](#frame-the-problem-goals--metrics)
4. [Collect & prepare the data](#collect--prepare-the-data)
5. [How to explore, extract aur engineer features](#how-to-explore-extract-aur-engineer-features)
    * [Code Example: Mutual Information](#code-example-mutual-information)
6. [Select an ML algorithm](#select-an-ml-algorithm)
7. [Design aur tune the model](#design-aur-tune-the-model)
    * [Code Example: Bias-Variance Trade-Off](#code-example-bias-variance-trade-off)
8. [How to use cross-validation ke liye model selection](#how-to-use-cross-validation-ke liye-model-selection)
    * [Code Example: How to implement cross-validation in Python](#code-example-how-to-implement-cross-validation-in-python)
9. [Parameter tuning ke saath scikit-learn](#parameter-tuning-ke saath-scikit-learn)
    * [Code Example: Learning and Validation curves with yellowbricks](#code-example-learning-and-validation-curves-with-yellowbricks)
    * [Code Example: Parameter tuning using GridSearchCV and pipeline](#code-example-parameter-tuning-using-gridsearchcv-and-pipeline)
10. [Challenges ke saath cross-validation mein finance](#challenges-ke saath-cross-validation-mein-finance)
    * [Purging, embargoing, and combinatorial CV](#purging-embargoing-and-combinatorial-cv)


## How machine learning from data works

Many definitions ka ML revolve around the automated detection ka meaningful patterns mein data. Two prominent examples include:
- AI pioneer Arthur Samuelson defined ML mein 1959 as a subfield ka computer science that gives computers the ability to learn without being explicitly programmed. 
- Tom Mitchell, one ka the current leaders mein the field, pinned down a well-posed learning problem more specifically mein 1998: a computer program learns from experience ke saath respect to a task aur a performance measure ka whether the performance ka the task improves ke saath experience (Mitchell, 1997).

Experience hai presented to an algorithm mein the form ka training data. The principal difference to previous attempts at building machines that solve problems hai that the rules that an algorithm use karta hai to make decisions hain learned from the data as opposed to being programmed by humans as was the case, ke liye example, ke liye expert systems prominent mein the 1980s.

Recommended textbooks that cover a wide range ka algorithms aur general applications include 
- [An Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/), James et al (2013)
- [The Elements ka Statistical Learning: Data Mining, Inference, aur Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/), Hastie, Tibshirani, aur Friedman (2009)
- [Pattern Recognition aur Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-aur-Machine-Learning-2006.pdf), Bishop (2006)
- [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html), Mitchell (1997).

### The key challenge: Finding the right algorithm ke liye the given task

The key challenge ka automated learning hai to identify patterns mein the training data that hain meaningful when generalizing the model's learning to new data. There hain a large number ka potential patterns that a model could identify, while the training data only constitutes a sample ka the larger set ka phenomena that the algorithm may encounter when performing the task mein the future. 

### Supervised Learning: teaching a task by example

Supervised learning hai the most commonly used type ka ML. hum will dedicate most ka the chapters mein this book to applications mein this category. The term supervised implies the presence ka an outcome variable that guides the learning process—that hai, it teaches the algorithm the correct solution to the task at hand. Supervised learning aims to capture a functional input-output relationship from individual samples that reflect this relationship aur to apply its learning by making valid statements about new data.

### Unsupervised learning: Exploring data to identify useful patterns

When solving an unsupervised learning problem, hum only observe the features aur have no measurements ka the outcome. Instead ka predicting future outcomes or inferring relationships among variables, unsupervised algorithms aim to identify structure mein the input that permits a new representation ka the information contained mein the data. 

#### Use cases ke liye trading strategies: From risk management to text processing
There hain numerous trading use cases ke liye unsupervised learning that hum will cover mein later chapters:
- Grouping together securities ke saath similar risk aur return characteristics (see [hierarchical risk parity mein Chapter 13](../13_unsupervised_learning/04_hierarchical_risk_parity)
- Finding a small number ka risk factors driving the performance ka a much larger number ka securities use karke [principal component analysis](../13_unsupervised_learning/01_linear_dimensionality_reduction)) or autoencoders ([Chapter 20](../20_autoencoders_for_conditional_risk_factors)
- Identifying latent topics mein a body ka documents (ke liye example, earnings call transcripts) that comprise the most important aspects ka those documents ([Chapter 15](../15_topic_modeling))

### Reinforcement learning: Learning by doing, one step at a time

Reinforcement learning (RL) hai the third type ka ML. It centers on an agent that needs to pick an action at each time step based on information provided by the environment. The agent could be a self-driving car, a program playing a board game or a video game, or a trading strategy operating mein a certain security market. 

You find an excellent introduction mein [Sutton aur Barto](http://www.incompleteideas.net/book/the-book-2nd.html), 2018.

## The Machine Learning Workflow

Developing an ML solution requires a systematic approach to maximize the chances ka success while proceeding efficiently. It hai also important to make the process transparent aur replicable to facilitate collaboration, maintenance, aur subsequent refinements.

The process hai iterative throughout, aur the effort at different stages will vary according to the project. Nonethelesee, this process should generally include the following steps:

1. Frame the problem, identify a target metric, aur define success
2. Source, clean, aur validate the data
3. Understand your data aur generate informative features
4. Pick one or more machine learning algorithms suitable ke liye your data
5. Train, test, aur tune your models
6. Use your model to solve the original problem

### Code Example: ML workflow ke saath K-nearest neighbors

Notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb) contain karta hai several examples that illustrate the machine learning workflow use karke a simple dataset ka house prices.

- sklearn [Documentation](http://scikit-learn.org/stable/documentation.html)
- k-nearest neighbors [tutorial](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn) aur [visualization](http://vision.stanford.edu/teaching/cs231n-demos/knn/)

## Frame the problem: goals & metrics

The starting point ke liye any machine learning exercise hai the ultimate use case it aims to address. Sometimes, this goal will be statistical inference mein order to identify an association between variables or even a causal relationship. Most frequently, however, the goal will be the direct prediction ka an outcome to yield a trading signal.

## Collect & prepare the data

hum addressed the sourcing ka market aur fundamental data mein [Chapter 2](../02_market_and_fundamental_data), aur ke liye alternative data mein [Chapter 3](../03_alternative_data). hum will continue to work ke saath various examples ka these sources as hum illustrate the application ka the various models mein later chapters. 

## How to explore, extract aur engineer features

Understanding the distribution ka individual variables aur the relationships among outcomes aur features hai the basis ke liye picking a suitable algorithm. Yeh typically starts ke saath visualizations such as scatter plots, as illustrated mein the companion notebook (aur shown mein the following image), but also includes numerical evaluations ranging from linear metrics, such as the correlation, to nonlinear statistics, such as the Spearman rank correlation coefficient that hum encountered when hum introduced the information coefficient. It also includes information-theoretic measures, such as mutual information

### Code Example: Mutual Information

Notebook [mutual_information](02_mutual_information.ipynb) applies information theory to the financial data hum created mein the notebook [feature_engineering](../04_alpha_factor_research/00_data/feature_engineering.ipynb), mein the chapter [Alpha Factors – Research aur Evaluation]((../04_alpha_factor_research).

## Select an ML algorithm

The remainder ka this book will introduce several model families, ranging from linear models, which make fairly strong assumptions about the nature ka the functional relationship between input aur output variables, to deep neural networks, which make very few assumptions.

## Design aur tune the model

The ML process includes steps to diagnose aur manage model complexity based on estimates ka the model's generalization error. An unbiased estimate requires a statistically sound aur efficient procedure, as well as error metrics that align ke saath the output variable type, which also determines whether hum hain dealing ke saath a regression, classification, or ranking problem.

### Code Example: Bias-Variance Trade-Off

The errors that an ML model makes when predicting outcomes ke liye new input data can be broken down into reducible aur irreducible parts. The irreducible part hai due to random variation (noise) mein the data that hai not measured, such as relevant but missing variables or natural variation. 

Notebook [bias_variance](03_bias_variance.ipynb) demonstrate karta hai overfitting by approximating a cosine function use karke increasingly complex polynomials aur measuring the mein-sample error.  It draws 10 random samples ke saath some added noise (n = 30) to learn a polynomial ka varying complexity. Each time, the model predicts new data points aur hum capture the mean-squared error ke liye these predictions, as well as the standard deviation ka these errors. It goes on to illustrate the impact ka overfitting versus underfitting by trying to learn a Taylor series approximation ka the cosine function ka ninth degree ke saath some added noise. mein the following diagram, hum draw random samples ka the true function aur fit polynomials that underfit, overfit, aur provide an approximately correct degree ka flexibility.

## How to use cross-validation ke liye model selection

When several candidate models (that hai, algorithms) hain available ke liye your use case, the act ka choosing one ka them hai called the model selection problem. Model selection aims to identify the model that will produce the lowest prediction error given new data.

### Code Example: How to implement cross-validation mein Python

The script [cross_validation](04_cross_validation.py) illustrates various options ke liye splitting data into training aur test sets by showing how the indices ka a mock dataset ke saath ten observations hain assigned to the train aur test set.
 
## Parameter tuning ke saath scikit-learn

Model selection typically involves repeated cross-validation ka the out-ka-sample performance ka models use karke different algorithms (such as linear regression aur random forest) or different configurations. Different configurations may involve changes to hyperparameters or the inclusion or exclusion ka different variables.

### Code Example: Learning aur Validation curves ke saath yellowbricks

Notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb)) demonstrate karta hai the use ka learning aur validation  illustrates the use ka various model selection techniques. 

- Yellowbrick: Machine Learning Visualization [docs](http://www.scikit-yb.org/en/latest/)

### Code Example: Parameter tuning use karke GridSearchCV aur pipeline

Since hyperparameter tuning hai a key ingredient ka the machine learning workflow, there hain tools to automate this process. The sklearn library includes a GridSearchCV interface that cross-validates all combinations ka parameters mein parallel, captures the result, aur automatically trains the model use karke the parameter setting that performed best during cross-validation on the full dataset.

mein practice, the training aur validation sets often require some processing prior to cross-validation. Scikit-learn offers the Pipeline to also automate any requisite feature-processing steps mein the automated hyperparameter tuning facilitated by GridSearchCV.

The implementation examples mein the included machine_learning_workflow.ipynb notebook to see these tools mein action.

Notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb)) also demonstrate karta hai the use ka these tools.

## Challenges ke saath cross-validation mein finance

A key assumption ke liye the cross-validation methods discussed so far hai the independent aur identical (iid) distribution ka the samples available ke liye training.
ke liye financial data, this hai often not the case. On the contrary, financial data hai neither independently nor identically distributed because ka serial correlation aur time-varying standard deviation, also known as heteroskedasticity

### Purging, embargoing, aur combinatorial CV

ke liye financial data, labels hain often derived from overlapping data points as returns hain computed from prices mein multiple periods. mein the context ka trading strategies, the results ka a model's prediction, which may imply taking a position mein an asset, may only be known later, when this decision hai evaluated—ke liye example, when a position hai closed out. 

The resulting risks include the leaking ka information from the test into the training set, likely leading to an artificially inflated performance that needs to be addressed by ensuring that all data hai point-mein-time—that hai, truly available aur known at the time it hai used as the input ke liye a model. Several methods have been proposed by Marcos Lopez de Prado mein [Advances mein Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) to address these challenges ka financial data ke liye cross-validation:

- Purging: Eliminate training data points where the evaluation occurs after the prediction ka a point-mein-time data point mein the validation set to avoid look-ahead bias.
- Embargoing: Further eliminate training samples that follow a test period.
