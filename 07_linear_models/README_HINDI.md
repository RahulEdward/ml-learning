# Linear Models: Risk Factors se Asset Return Forecasts tak

Linear models ki family sabse useful hypothesis classes mein se ek represent karti hai. Algorithmic trading mein widely apply kiye jane wale kai learning algorithms linear predictors par bharosa karte hain kyunki unhe efficiently train kiya ja sakta hai, noisy financial data ke prati relatively robust hote hain aur finance ki theory se strong links rakhte hain. Linear predictors intuitive bhi hote hain, interpret karne mein aasan hote hain, aur aksar data ko reasonably well fit karte hain ya kam se kam ek accha baseline provide karte hain.

Linear regression 200 saal se zyada samay se jana jata hai jab se Legendre aur Gauss ne ise astronomy par apply kiya aur iski statistical properties analyze karna shuru kiya. Tab se numerous extensions ne linear regression model aur uske parameters seekhne ke liye baseline ordinary least squares (OLS) method ko adapt kiya hai:

- **Generalized linear models** (GLM) un response variables ki anumati dekar applications ka scope badhate hain jo normal distribution ke alawa koi aur error distribution imply karte hain. GLMs mein categorical response variables ke liye probit ya logistic models shamil hain jo classification problems mein dikhayi dete hain.
- Zyada **robust estimation methods** wahan statistical inference ki suvidha dete hain jahan data baseline assumptions ko violate karta hai, example ke liye, samay ke saath correlation ya observations ke aar-paar. Ye aksar panel data ke saath hota hai jisme same units par repeated observations hote hain jaise assets ke universe par historical returns.
- **Shrinkage methods** linear models ki predictive performance improve karne ka maksad rakhte hain. Wo ek complexity penalty use karte hain jo model ke variance ko kam karne aur out-of-sample predictive performance improve karne ke goal ke saath model dwara seekhe gaye coefficients ko bias karta hai.

Practice mein, linear models inference aur prediction ke goals ke saath regression aur classification problems par apply kiye jate hain. Academic aur industry researchers dwara numerous asset pricing models develop kiye gaye hain jo linear regression ka labh uthate hain. Applications mein significant factors ki pehchan shamil hai jo better risk aur performance management ke liye asset returns drive karte hain, saath hi various time horizons par returns ka prediction bhi. Dusri taraf, classification problems mein directional price forecasts shamil hain. Is chapter mein, hum nimnlikhit vishayon ko cover karenge:

## Vishay Soochi (Content)

1. [Linear regression: Inference se prediction tak](#linear-regression-inference-se-prediction-tak)
2. [Baseline model: Multiple linear regression](#baseline-model-multiple-linear-regression)
    * [Code Example: `statsmodels` aur `scikit-learn` ke saath Simple aur multiple linear regression](#code-example-statsmodels-aur-scikit-learn-ke-saath-simple-aur-multiple-linear-regression)
3. [Linear factor model kaise banayein](#linear-factor-model-kaise-banayein)
    * [CAPM se Fama—French five-factor model tak](#capm-se-famafrench-five-factor-model-tak)
    * [Risk factors prapt karna](#risk-factors-prapt-karna)
    * [Code Example: Fama—Macbeth regression](#code-example-famamacbeth-regression)
4. [Shrinkage methods: Linear regression ke liye Regularization](#shrinkage-methods-linear-regression-ke-liye-regularization)
    * [Overfitting ke khilaf Hedging – linear models mein regularization](#overfitting-ke-khilaf-hedging-linear-models-mein-regularization)
    * [Ridge regression](#ridge-regression)
    * [Lasso regression](#lasso-regression)
5. [Linear regression ke saath stock returns kaise predict karein](#linear-regression-ke-saath-stock-returns-kaise-predict-karein)
    * [Code Examples: stock returns ke liye inference aur prediction](#code-examples-stock-returns-ke-liye-inference-aur-prediction)
6. [Linear classification](#linear-classification)
    * [Logistic regression model](#logistic-regression-model)
    * [Code Example: statsmodels ke saath inference kaise conduct karein](#code-example-statsmodels-ke-saath-inference-kaise-conduct-karein)
    * [Code examples: prediction ke liye logistic regression ka use kaise karein](#code-examples-prediction-ke-liye-logistic-regression-ka-use-kaise-karein)
7. [References](#references)

## Linear regression: Inference se prediction tak

Ye section linear models ke liye baseline cross-section aur panel techniques introduce karta hai aur important enhancements jo accurate estimates produce karte hain jab key assumptions violate ho jate hain. Ye in methods ko factor models (jo algorithmic trading strategies ke development mein ubiquitous hain) estimate karke illustrate karna jari rakhta hai. Ant mein, ye regularization methods par focus karta hai.

- [Introductory Econometrics](http://economics.ut.ac.ir/documents/3030266/14100645/Jeffrey_M._Wooldridge_Introductory_Econometrics_A_Modern_Approach__2012.pdf), Wooldridge, 2012

## Baseline model: Multiple linear regression

Ye section model ke specification aur objective function, uske parameters seekhne ke methods, statistical assumptions jo inference aur in assumptions ke diagnostics ki anumati dete hain, saath hi model ko un situations mein adapt karne ke liye extensions introduce karta hai jahan ye assumptions fail ho jate hain. Content mein shamil hain:

- Model ko kaise formulate aur train karein
- Gauss-Markov Theorem
- Statistical inference kaise conduct karein
- Problems ko kaise diagnose aur remedy karein
- Practice mein linear regression kaise run karein

### Code Example: `statsmodels` aur `scikit-learn` ke saath Simple aur multiple linear regression

Notebook [linear_regression_intro](01_linear_regression_intro.ipynb) simple aur multiple linear regression model demonstrate karta hai, baad wala `statsmodels` aur `scikit-learn` par based OLS aur gradient descent dono ka use karta hai.

## Linear factor model kaise banayein

Algorithmic trading strategies asset ke return aur risk ke un sources (jo in returns ke main drivers represent karte hain) ke beech relationship ko quantify karne ke liye linear factor models ka use karti hain. Har factor risk ek premium carry karta hai, aur total asset return ke in risk premia ke weighted average ke correspond hone ki umeed ki ja sakti hai.

### CAPM se Fama—French five-factor model tak

Risk factors quantitative models ke lie ek key ingredient rahe hain jab se Capital Asset Pricing Model (CAPM) ne ek single factor, risk-free rate ke upar overall market ke expected excess return ke respective exposure ka use karke sabhi assets ke expected returns explain kiye.

Ye classic fundamental analysis (a la Dodd aur Graham) se alag hai jahan returns firm characteristics par depend karte hain. Rationale ye hai ki, aggregate mein, investors diversification ke zariye is so-called systematic risk ko eliminate nahi kar sakte. Isliye, equilibrium mein, unhe apni systematic risk ke anurup asset hold karne ke liye compensation ki zarurat hoti hai. Model imply karta hai ki, efficient markets ko dekhte huye jahan prices turant sabhi public information reflect karti hain, koi superior risk-adjusted returns nahi hone chahiye.

### Risk factors prapt karna

[Fama—French risk factors](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) un diversified portfolios par return difference ke roop mein compute kiye jate hain jinki values given risk factor ko reflect karne wale metrics ke hisab se high ya low hoti hain. Ye returns stocks ko in metrics ke hisab se sort karke aur fir certain percentile se upar stocks ko long karke jabki certain percentile se niche stocks ko short karke prapt kiye jate hain. Risk factors se jude metrics nimnlikhit define kiye gaye hain:

- Size: Market Equity (ME)
- Value: Book Value of Equity (BE) divided by ME
- Operating Profitability (OP): Revenue minus cost of goods sold/assets
- Investment: Investment/assets

Fama aur French updated risk factor aur research portfolio data apni [website]((http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)) ke madhyam se available karate hain, aur aap data prapt karne ke liye [pandas_datareader](https://pandas-datareader.readthedocs.io/en/latest/) library ka use kar sakte hain.

### Code Example: Fama—Macbeth regression

Residuals ke correlation se hone wali inference problem ko address karne ke liye, Fama aur MacBeth ne factors par returns ke cross-sectional regression ke liye ek two-step methodology propose kiya. Two-stage Fama—Macbeth regression market dwara kisi particular risk factor ke exposure ke liye reward kiye gaye premium ko estimate karne ke liye design kiya gaya hai. Two stages mein shamil hain:
- **First stage**: N time-series regression, har asset ya portfolio ke liye ek, factor loadings estimate karne ke liye factors par iske excess returns ka.
- **Second stage**: T cross-sectional regression, har time period ke liye ek, risk premium estimate karne ke liye.

Notebook [fama_macbeth](02_fama_macbeth.ipynb) illustrate karta hai ki Fama-Macbeth regression kaise run karein, jisme [LinearModels](https://bashtage.github.io/linearmodels/doc/) library ka use shamil hai.

## Shrinkage methods: Linear regression ke liye Regularization

Jab ek linear regression model mein kai correlated variables hote hain, to unke coefficients poorly determined honge kyunki RSS par ek large positive coefficient ka effect correlated variable par similarly large negative coefficient dwara cancel kiya ja sakta hai. Isliye, model mein coefficients ke is wiggle room ki wajah se high variance ki tendency hogi jo is risk ko badhata hai ki model sample par overfit karega.

### Overfitting ke khilaf Hedging – regularization in linear models

Overfitting ko control karne ke liye ek popular technique regularization hai, jisme coefficients ko large values tak pahunchne se discourage karne ke liye error function mein ek penalty term add karna shamil hai. Dusre shabdon mein, coefficients par size constraints out-of-sample predictions par hone wale potentially negative impact ko kam kar sakte hain. Hum sabhi models ke liye regularization methods ka saamna karenge kyunki overfitting ek aisi pervasive problem hai.

Is section mein, hum shrinkage methods introduce karenge jo ab tak charcha kiye gaye linear models ke approaches par improve karne ke liye do motivations address karte hain:
- Prediction accuracy: Least squares estimates ka low bias lekin high variance suggest karta hai ki generalization error ko kuch coefficients ko shrink karke ya zero set karke kam kiya ja sakta hai, jisse model ke variance mein kami ke liye slightly higher bias trade off kiya ja sakta hai.
- Interpretation: Badi sankhya mein predictors results ki big picture ke interpretation ya communication ko complicate kar sakte hain. Model ko strongest effects wale parameters ke smaller subset tak limit karne ke liye kuch detail sacrifice karna preferable ho sakta hai.

### Ridge regression

Ridge regression objective function mein ek penalty add karke regression coefficients ko shrink karta hai jo squared coefficients ke sum ke barabar hoti hai, jo badle mein coefficient vector ke L2 norm ke correspond hoti hai.

### Lasso regression

Lasso (jise signal processing mein basis pursuit ke roop mein jana jata hai) residuals ke squares ke sum mein penalty add karke bhi coefficients ko shrink karta hai, lekin lasso penalty ka effect thoda alag hota hai. Lasso penalty coefficient vector ki absolute values ka sum hai, jo iske L1 norm ke correspond hoti hai.

## Linear regression ke saath stock returns kaise predict karein

Is section mein, hum returns predict karne aur trading signals generate karne ke liye shrinkage ke saath aur bina shrinkage ke linear regression ka use karenge. Is maksad ke liye, hum pehle ek dataset create karte hain aur fir statsmodels aur sklearn ke saath unka usage illustrate karne ke liye pichle section mein charcha kiye gaye linear regression models apply karte hain.

### Code Examples: stock returns ke liye inference aur prediction

- Notebook [preparing_the_model_data](03_preparing_the_model_data.ipynb) US equities ka ek universe select karta hai aur daily returns predict karne ke liye kai features create karta hai.
- Notebook [statistical_inference_of_stock_returns_with_statsmodels](04_statistical_inference_of_stock_returns_with_statsmodels.ipynb) OLS aur `statsmodels` library ka use karke kai linear regression models estimate karta hai.
- Notebook [predicting_stock_returns_with_linear_regression](05_predicting_stock_returns_with_linear_regression.ipynb) dikhata hai ki linear regression ke saath-saath `scikit-klearn` ke saath ridge aur lasso models ka use karke daily stock return kaise predict karein.
- Notebook [evaluating_signals_using_alphalens](06_evaluating_signals_using_alphalens.ipynb) `alphalens` ka use karke model predictions evaluate karta hai.

## Linear classification

Qualitative response predict karne ke liye kai alag-alag classification techniques hain. Is section mein, hum widely used logistic regression introduce karenge jo linear regression se closely related hai. Hum agle chapters mein zyada complex methods address karenge, generalized additive models par jisme decision trees aur random forests shamil hain, saath hi gradient boosting machines aur neural networks.

### Logistic regression model

Logistic regression model output classes ki probabilities model karne ki iccha se uthta hai jabki ek function diya gaya ho jo x mein linear ho, bilkul linear regression model ki tarah, jabki saath hi ye ensure karta ho ki wo one sum hon aur [0, 1] mein rahein jaisa ki hum probabilities se umeed karenge.

Is section mein, hum logistic regression model ka objective aur functional form introduce karte hain aur training method describe karte hain. Fir hum illustrate karte hain ki statsmodels ka use karke macro data ke saath statistical inference ke liye logistic regression ka use kaise karein, aur sklearn dwara implemented regularized logistic regression ka use karke price movements kaise predict karein.

### Code Example: statsmodels ke saath inference kaise conduct karein

Notebook [logistic_regression_macro_data](07_logistic_regression_macro_data.ipynb)` illustrate karta hai ki macro data par logistic regression kaise run karein aur [statsmodels](https://www.statsmodels.org/stable/index.html) ka use karke statistical inference kaise conduct karein.

### Code examples: prediction ke liye logistic regression ka use kaise karein

Lasso L1 penalty aur ridge L2 penalty dono ka use logistic regression ke saath kiya ja sakta hai. Unka wahi shrinkage effect hota hai jaisa humne abhi charcha ki hai, aur lasso ko kisi bhi linear regression model ke saath variable selection ke liye fir se use kiya ja sakta hai.

Jaise linear regression ke saath, input variables ko standardize karna mahatvapurn hai kyunki regularized models scale sensitive hote hain. Regularization hyperparameter ko linear regression case ki tarah cross-validation ka use karke tuning ki bhi zarurat hoti hai.

Notebook [predicting_price_movements_with_logistic_regression](08_predicting_price_movements_with_logistic_regression.ipynb) demonstrate karta hai ki stock price movement prediction ke liye Logistic Regression ka use kaise karein.

## References

- [Risk, Return, and Equilibrium: Empirical Tests](https://www.jstor.org/stable/1831028), Eugene F. Fama and James D. MacBeth, Journal of Political Economy, 81 (1973), pp. 607–636
- [Asset Pricing](http://faculty.chicagobooth.edu/john.cochrane/teaching/asset_pricing.htm), John Cochrane, 2001
