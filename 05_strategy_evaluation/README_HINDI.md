# Portfolio Optimization aur Performance Evaluation

Market conditions ke tehat implementation se pehle kisi strategy ko test karne ke liye, humein un trades ko simulate karne ki zarurat hoti hai jo algorithm karega aur unki performance ko verify karna hota hai. Strategy evaluation mein strategy ke parameters ko optimize karne ke liye historical data ke khilaf backtesting shamil hai, aur naye, out-of-sample data ke khilaf in-sample performance ko validate karne ke liye forward-testing shamil hai. Iska maksad strategy ko specific past circumstances (purani paristhitiyon) ke hisab se tailor karne se hone wali false discoveries se bachna hai.

Portfolio context mein, positive asset returns negative price movements ko offset (santulit) kar sakte hain. Ek asset ke liye positive price changes dusre asset par hone wale losses ko offset karne ki sambhavna utni hi zyada rakhte hain jitna kam unn dono positions ke beech correlation hoga. Portfolio risk positions ki covariance par kaise nirbhar karta hai, is aadhar par Harry Markowitz ne 1952 mein diversification par based modern portfolio management ki theory develop ki. Iska natija mean-variance optimization hai jo risk ko minimize karne ke liye assets ke given set ke liye weights select karta hai, jise given expected return ke liye returns ke standard deviation ke roop mein measure kiya jata hai.

Capital asset pricing model (CAPM) ek risk premium introduce karta hai, jise risk-free investment se excess expected return ke roop mein measure kiya jata hai, jo asset hold karne ke liye ek equilibrium reward hai. Ye reward ek single risk factor—market—ko exposure dene ka compensation hai jo asset ke liye idiosyncratic hone ke bajaye systematic hai aur isliye ise diversify karke hataya nahi ja sakta.

Jaise-jaise additional risk factors aur exposure ke liye zyada granular choices ubhar kar aayi hain, Risk management develop hokar aur sophisticated ho gaya hai. Kelly criterion dynamic portfolio optimization ke liye ek popular approach hai, jo samay ke saath positions ke sequence ka choice hai; ise Edward Thorp ne 1968 mein gambling mein iske original application se stock market ke liye mashhur taur par adapt kiya tha.

Natijan, portfolios ko optimize karne ke liye kai approaches hain jinme assets ke beech hierarchical relationships seekhne aur unki holdings ko portfolio risk profile ke sandarbh mein complements ya substitutes maanne ke liye machine learning (ML) ka application shamil hai. Ye chapter nimnlikhit vishayon ko cover karega:

## Vishay Soochi (Content)

1. [Portfolio performance ko kaise measure karein](#portfolio-performance-ko-kaise-measure-karein)
    * [(Adjusted) Sharpe Ratio](#adjusted-sharpe-ratio)
    * [Active management ka fundamental law](#active-management-ka-fundamental-law)
2. [Portfolio Risk aur Return ko kaise manage karein](#portfolio-risk-aur-return-ko-kaise-manage-karein)
    * [Modern portfolio management ka evolution](#modern-portfolio-management-ka-evolution)
    * [Mean-variance optimization](#mean-variance-optimization)
        - [Code Examples: Python mein efficient frontier dhundhna](#code-examples-python-mein-efficient-frontier-dhundhna)
    * [Mean-variance optimization ke vikalp (Alternatives)](#mean-variance-optimization-ke-vikalp)
        - [1/N portfolio](#1n-portfolio)
        - [Minimum-variance portfolio](#minimum-variance-portfolio)
        - [Black-Litterman approach](#black-litterman-approach)
        - [Apne bets ka size kaise tay karein – Kelly rule](#apne-bets-ka-size-kaise-tay-karein-kelly-rule)
        - [Python ke saath MV Optimization ke vikalp](#python-ke-saath-mv-optimization-ke-vikalp)
    * [Hierarchical Risk Parity](#hierarchical-risk-parity)
3. [`Zipline` ke saath portfolio ki Trading aur managing](#zipline-ke-saath-portfolio-ki-trading-aur-managing)
    * [Code Examples: Trades aur portfolio optimization ke saath Backtests](#code-examples-trades-aur-portfolio-optimization-ke-saath-backtests)
4. [`pyfolio` ke saath backtest performance measure karein](#pyfolio-ke-saath-backtest-performance-measure-karein)
    * [Code Example: `Zipline` backtest se `pyfolio` evaluation](#code-example-zipline-backtest-se-pyfolio-evaluation)

## Portfolio performance ko kaise measure karein

Alag-alag strategies ko evaluate aur compare karne ke liye ya existing strategy ko improve karne ke liye, humein aise metrics ki zarurat hoti hai jo hamare objectives ke hisab se unki performance reflect karein. Investment aur trading mein, sabse common objectives **investment portfolio ka return aur risk** hain.

Return aur risk objectives ek trade-off imply karte hain: zyada risk lene se kuch halaat mein higher returns mil sakte hain, lekin iska matlab zyada downside bhi hai. Alag-alag strategies is trade-off ko kaise navigate karti hain ise compare karne ke liye, wo ratios bahut popular hain jo per unit of risk return ka measure compute karte hain. Hum baari-baari se **Sharpe ratio** aur **information ratio** (IR) par charcha karenge.

### (Adjusted) Sharpe Ratio

Ex-ante Sharpe Ratio (SR) portfolio ke expected excess portfolio ko is excess return ki volatility se compare karta hai, jise iske standard deviation dwara measure kiya jata hai. Ye compensation ko per unit of risk taken average excess return ke roop mein measure karta hai. Ise data se estimate kiya ja sakta hai.

Financial returns aksar iid assumptions ko violate karte hain. Andrew Lo ne un returns ke liye distribution aur time aggregation mein zaruri adjustments derive kiye hain jo stationary to hain lekin autocorrelated hain. Ye zaruri hai kyunki investment strategies ki time-series properties (example ke liye, mean reversion, momentum, aur serial correlation ke anya forms) ka SR estimator par hi non-trivial impact pad sakta hai, khaas taur par jab higher-frequency data se SR ko annualize kiya ja raha ho.

- [The Statistics of Sharpe Ratios](https://www.jstor.org/stable/4480405?seq=1#page_scan_tab_contents), Andrew Lo, Financial Analysts Journal, 2002

### Active management ka fundamental law

Ye ek dilchasp tathya (fact) hai ki Renaissance Technologies (RenTec), Jim Simons dwara sthapit top-performing quant fund jiska zikra humne [Chapter 1](../01_machine_learning_for_trading) mein kiya tha, ne behad alag approaches ke bawajood Warren Buffet jaisa hi returns produce kiya hai. Warren Buffet ki investment firm Berkshire Hathaway karib 100-150 stocks ko kaafi lambe samay tak hold karti hai, jabki RenTec shayad pratidin 100,000 trades execute karti hai. Hum in alag-alag strategies ko kaise compare kar sakte hain?

ML objective functions ko optimize karne ke baare mein hai. Algorithmic trading mein, objectives overall investment portfolio ka return aur risk hote hain, jo aamtaur par kisi benchmark (jo cash, risk-free interest rate, ya S&P 500 jaisa asset price index ho sakta hai) ke relative hota hai.

Ek high Information Ratio (IR) liye gaye additional risk ke relative attractive out-performance imply karta hai. Fundamental Law of Active Management IR ko forecasting skill ke measure ke roop mein information coefficient (IC), aur independent bets ke zariye is skill ko apply karne ki kshamta mein torta hai. Ye aksar khelne (high breadth) aur accha khelne (high IC), dono ke mahatva ko summarize karta hai.

IC alpha factor aur uske signals se aane wale forward returns ke beech correlation measure karta hai aur manager ki forecasting skills ki accuracy ko capture karta hai. Strategy ki breadth ko ek given time period mein investor dwara lagaye gaye independent bets ki sankhya se measure kiya jata hai, aur dono values ka product IR ke proportional hota hai, jise appraisal risk (Treynor aur Black) bhi kaha jata hai.

Fundamental law zaruri hai kyunki ye outperformance ke key drivers ko highlight karta hai: accurate predictions aur independent forecasts karne aur in forecasts par act karne ki kshamta, dono mayne rakhte hain. Practice mein, forecasts ke beech cross-sectional aur time-series correlation ko dekhte huye strategy ki breadth estimate karna mushkil hai.

- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [How to Use Security Analysis to Improve Portfolio Selection](https://econpapers.repec.org/article/ucpjnlbus/v_3a46_3ay_3a1973_3ai_3a1_3ap_3a66-86.htm), Jack L Treynor and Fischer Black, Journal of Business, 1973
- [Portfolio Constraints and the Fundamental Law of Active Management](https://faculty.fuqua.duke.edu/~charvey/Teaching/BA491_2005/Transfer_coefficient.pdf), Clarke et al 2002

## Portfolio Risk aur Return ko kaise manage karein

Portfolio management ka maksad financial instruments mein un positions ko pick aur size karna hota hai jo kisi benchmark ke regarding desired risk-return trade-off hasil karein. Ek portfolio manager ke roop mein, har period mein, aap wo positions select karte hain jo target return hasil karte huye risks kam karne ke liye diversification ko optimize karein. Periods ke aar-paar, in positions ko target risk profile hasil karne ya banaye rakhne ke liye price movements se hone wale weights mein badlav ko account karne ke liye rebalancing ki zarurat ho sakti hai.

### Modern portfolio management ka evolution

Diversification humein ye exploit karke given expected return ke liye risks kam karne ki suvidha deta hai ki kaise imperfect correlation ek asset ke gains ko dusre asset ke losses ki bharpayi karne deta hai. Harry Markowitz ne 1952 mein modern portfolio theory (MPT) ka aavishkar kiya aur appropriate portfolio weights chunkar diversification ko optimize karne ke liye mathematical tools pradan kiye.
 
### Mean-variance optimization

Modern portfolio theory given expected return ke liye volatility minimize karne, ya given level of volatility ke liye returns maximize karne ke liye optimal portfolio weights solve karti hai. Key requisite inputs expected asset returns, standard deviations, aur covariance matrix hain.

- [Portfolio Selection](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf), Harry Markowitz, The Journal of Finance, 1952
- [The Capital Asset Pricing Model: Theory and Evidence](http://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S6B-FamaFrench-CAPM-JEP04.pdf), Eugene F. Fama and Kenneth R. French, Journal of Economic Perspectives, 2004

#### Code Examples: Python mein efficient frontier dhundhna

Hum scipy.optimize.minimize aur asset returns, standard deviations, aur covariance matrix ke liye historical estimates ka use karke ek efficient frontier calculate kar sakte hain.
- Python mein efficient frontier compute karne ke liye notebook [mean_variance_optimization](04_mean_variance_optimization.ipynb) dekhein.

### Mean-variance optimization ke vikalp (Alternatives)

Mean-variance optimization problem ke liye accurate inputs ke saath challenges ne kai practical alternatives ko apnane ko badhava diya hai jo mean, variance, ya dono ko constrain karte hain, ya return estimates ko omit kar dete hain jo zyada challenging hote hain, jaise risk parity approach jis par hum is section mein baad mein charcha karenge.

#### 1/N portfolio

Simple portfolios complex models ke added value ko mapne (gauge) ke liye useful benchmarks pradan karte hain jo overfitting ka risk generate karte hain. Sabse simple strategy—ek equally-weighted portfolio—ko best performers mein se ek dikhaya gaya hai.

#### Minimum-variance portfolio

Ek aur alternative global minimum-variance (GMV) portfolio hai, jo risk ke minimization ko prathmikta deta hai. Ise efficient frontier figure mein dikhaya gaya hai aur mean-variance framework ka use karke portfolio standard deviation ko minimize karke ise calculate kiya ja sakta hai.

#### Black-Litterman approach

Black aur Litterman (1992) ka Global Portfolio Optimization approach economic models ko statistical learning ke saath combine karta hai aur popular hai kyunki ye expected returns ke estimates generate karta hai jo kai situations mein plausible (tark-sangt) hote hain.
Ye technique maanti hai ki market ek mean-variance portfolio hai jaisa ki CAPM equilibrium model dwara imply kiya gaya hai. Ye is tathya par banta hai ki observed market capitalization ko market dwara har security ko assign kiye gaye optimal weights ke roop mein mana ja sakta hai. Market weights market prices ko reflect karte hain jo, badle mein, future returns ki market expectations ko embody karte hain.

- [Global Portfolio Optimization](http://www.sef.hku.hk/tpg/econ6017/2011/black-litterman-1992.pdf), Black, Fischer; Litterman, Robert
Financial Analysts Journal, 1992

#### Apne bets ka size kaise tay karein – Kelly rule

Kelly rule ka gambling mein lamba itihas hai kyunki ye guidance deta hai ki terminal wealth ko maximize karne ke liye bets ke ek (infinite) sequence par varying (lekin favorable) odds ke saath kitna daanv (stake) lagana hai. Ise 1956 mein John Kelly dwara A New Interpretation of the Information Rate ke roop mein publish kiya gaya tha jo Bell Labs mein Claude Shannon ke colleague the. Wo naye quiz show The $64,000 Question mein candidates par lagaye gaye bets se intrigued the, jahan west coast par ek viewer ne winners ke baare mein insider information prapt karne ke liye teen ghante ki deri (delay) ka use kiya.

Kelly ne Shannon ki information theory se connection joda taaki us bet ko solve kiya ja sake jo long-term capital growth ke liye optimal hai jab odds favorable hon, lekin uncertainty bani rahe. Unka rule har game ki success ke odds ke function ke roop mein logarithmic wealth ko maximize karta hai, aur isme implicit bankruptcy protection shamil hai kyunki log(0) negative infinity hai taaki ek Kelly gambler naturally sab kuch khone se bachega.

- [A New Interpretation of Information Rate](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf), John Kelly, 1956

#### Python ke saath MV Optimization ke vikalp

- Notebook [kelly_rule](05_kelly_rule.ipynb) single aur multiple asset case ke liye application demonstrate karta hai.
- Baad wala result notebook [mean_variance_optimization](04_mean_variance_optimization.ipynb) mein bhi shamil hai, kai anya alternative approaches ke saath.

### Hierarchical Risk Parity

[Marcos Lopez de Prado](http://www.quantresearch.org/) dwara develop kiya gaya ye novel approach aamtaur par quadratic optimizers, aur khaas taur par Markowitz ke critical line algorithm (CLA) ki teen badi chintaon ko address karne ka maksad rakhta hai:
- instability (asthirta),
- concentration (ekagrata), aur
- underperformance.

Hierarchical Risk Parity (HRP) covariance matrix mein contained information ke aadhar par diversified portfolio banane ke liye graph theory aur machine-learning apply karta hai. Halaanki, quadratic optimizers ke vipreet, HRP ko covariance matrix ki invertibility ki zarurat nahi hoti. Vastav mein, HRP ill-degenerated ya singular covariance matrix par bhi portfolio compute kar sakta hai—jo quadratic optimizers ke liye namumkin kaam hai. Monte Carlo experiments dikhate hain ki HRP CLA se kam out-of-sample variance deliver karta hai, bhale hi minimum variance CLA ka optimization objective ho. HRP traditional risk parity methods ke mukable out of sample less risky portfolios bhi produce karta hai. Hum HRP par [Chapter 13](../13_unsupervised_learning) mein zyada vistar se charcha karenge jab hum trading ke liye unsupervised learning (hierarchical clustering sahit) ke applications par charcha karenge.

- [Building diversified portfolios that outperform out of sample](https://jpm.pm-research.com/content/42/4/59.short), Marcos López de Prado, The Journal of Portfolio Management 42, no. 4 (2016): 59-69.
- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Thomas Raffinot, 2016

Hum demonstrate karte hain ki HRP ko kaise implement karein aur ise Chapter 13 mein [Unsupervised Learning](../13_unsupervised_learning) mein alternatives se compare karte hain jahan hum hierarchical clustering bhi introduce karte hain.

## `Zipline` ke saath portfolio ki Trading aur managing

Open source [zipline](https://zipline.ml4trading.io/index.html) library ek event-driven backtesting system hai jise crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) dwara maintain aur production mein use kiya jata hai taaki algorithm-development aur live-trading aasan ho sake. Ye trade events par algorithm ke reaction ko automate karta hai aur ise current aur historical point-in-time data deta hai jo look-ahead bias se bachaata hai. [Chapter 8 - The ML4T Workflow](../08_strategy_workflow) mein `zipline` aur `backtrader` dono ka use karke backtesting ka zyada detailed, dedicated parichay hai.

[Chapter 4](../04_alpha_factor_research) mein, humne trailing cross-sectional market, fundamental, aur alternative data se alpha factors ke computation ko simulate karne ke liye `zipline` introduce kiya tha. Ab hum buy aur sell signals derive karne aur un par act karne ke liye alpha factors ka labh uthayenge.

### Code Examples: Trades aur portfolio optimization ke saath Backtests

Is section ke liye code nimnlikhit do notebooks mein rehta hai:
- Is section ke notebooks `conda` environment `backtest` use karte hain. Latest Docker image download karne ya apna environment set karne ke alternative tarikon ke liye kripya installation [instructions](../installation/README.md) dekhein.
- Notebook [backtest_with_trades](01_backtest_with_trades.ipynb) un trading decisions ko simulate karta hai jo Zipline ka use karke pichle chapter se simple MeanReversion alpha factor par based portfolio banate hain. Hum portfolio weights ko explicitly optimize nahi karte hain aur bas har holding ko equal value ki positions assign karte hain.
- Notebook [backtest_with_pf_optimization](02_backtest_with_pf_optimization.ipynb) demonstrate karta hai ki simple strategy backtest ke hisse ke roop mein PF optimization ka use kaise karein.

## `pyfolio` ke saath backtest performance measure karein

Pyfolio kai standard metrics ka use karke in-sample aur out-of-sample portfolio performance aur risk ke analysis ko aasan banata hai. Ye returns, positions, aur transactions ke analysis ko cover karne wale tear sheets produce karta hai, saath hi kai built-in scenarios ka use karke market stress ke periods ke dauran event risk, aur isme Bayesian out-of-sample performance analysis bhi shamil hai.

### Code Example: `Zipline` backtest se `pyfolio` evaluation

Notebook [pyfolio_demo](03_pyfolio_demo.ipynb) illustrate karta hai ki pichle folder mein conduct kiye gaye backtest se `pyfolio` input kaise extract karein. Phir ye `pyfolio` ka use karke kai performance metrics aur tear sheets calculate karne ki taraf badhta hai.

- Is notebook ke liye `conda` environment `backtest` ki zarurat hai. Latest Docker image chalane ya apna environment set karne ke alternative tarikon ke liye kripya [installation instructions](../installation/README.md) dekhein.
