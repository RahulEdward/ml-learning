# Machine Learning Workflow

Ye chapter is book ke part 2 ki shuruwat karta hai jahan hum illustrate karte hain ki aap trading ke liye supervised aur unsupervised machine learning (ML) models ki ek range ka use kaise kar sakte hain. Various Python libraries ka use karke relevant applications demonstrate karne se pehle hum har model ke assumptions aur use cases explain karenge. Models ki categories jinhe hum parts 2-4 mein cover karenge unme shamil hain:

- Cross-section, time series, aur panel data ke regression aur classification ke liye Linear models
- Generalized additive models, jisme nonlinear tree-based models shamil hain, jaise decision trees
- Ensemble models, jisme random forest aur gradient-boosting machines shamil hain
- Dimensionality reduction aur clustering ke liye Unsupervised linear aur nonlinear methods
- Neural network models, jisme recurrent aur convolutional architectures shamil hain
- Reinforcement learning models

Hum in models ko is book ke first part mein introduce kiye gaye market, fundamental, aur alternative data sources par apply karenge. Hum ab tak cover kiye gaye material par build karenge aur demonstrate karenge ki in models ko ek trading strategy mein kaise embed karein jo model signals ko trades mein translate karti hai, portfolio ko kaise optimize karein, aur strategy performance ko kaise evaluate karein.

Kai aspects hain jo inme se kai models aur unke applications mein common hain. Ye chapter in common aspects ko cover karta hai taaki hum agle chapters mein model-specific usage par focus kar sakein. Inme ek objective ya loss function ko optimize karke data se functional relationship seekhne ka overarching goal shamil hai. Inme model performance measure karne ke closely related methods bhi shamil hain.

Hum unsupervised aur supervised learning ke beech antar karte hain aur algorithmic trading ke liye use cases outline karte hain. Hum supervised regression aur classification problems, input aur output data ke beech relationships ke statistical inference ke liye supervised learning ka use aur future outputs ke prediction ke liye iske use ko contrast karte hain. Hum ye bhi illustrate karte hain ki kaise prediction errors model ke bias ya variance ki wajah se hote hain, ya data mein high noise-to-signal ratio ki wajah se. Sabse mahatvapurn baat, hum overfitting jaise errors ke sources detect karne aur aapke model ki performance improve karne ke methods present karte hain.

Agar aap pehle se hi ML se kaafi familiar hain, to feel free karke aage badhein aur sidhe ye seekhne mein dive karein ki algorithmic trading strategy ke liye alpha factors produce aur combine karne ke liye ML models ka use kaise karein.

## Vishay Soochi (Content)

1. [Data se machine learning kaise kaam karti hai](#data-se-machine-learning-kaise-kaam-karti-hai)
    * [Key challenge: Given task ke liye sahi algorithm dhundhna](#key-challenge-given-task-ke-liye-sahi-algorithm-dhundhna)
    * [Supervised Learning: example se task sikhana](#supervised-learning-example-se-task-sikhana)
    * [Unsupervised learning: Useful patterns pehchanne ke liye data explore karna](#unsupervised-learning-useful-patterns-pehchanne-ke-liye-data-explore-karna)
        - [Trading strategies ke liye use cases: Risk management se text processing tak](#trading-strategies-ke-liye-use-cases-risk-management-se-text-processing-tak)
    * [Reinforcement learning: Karke seekhna, ek baar mein ek kadam](#reinforcement-learning-karke-seekhna-ek-baar-mein-ek-kadam)
2. [Machine Learning Workflow](#machine-learning-workflow)
    * [Code Example: K-nearest neighbors ke saath ML workflow](#code-example-k-nearest-neighbors-ke-saath-ml-workflow)
3. [Problem frame karein: goals aur metrics](#problem-frame-karein-goals-aur-metrics)
4. [Data Collect aur prepare karein](#data-collect-aur-prepare-karein)
5. [Features ko kaise explore, extract aur engineer karein](#features-ko-kaise-explore-extract-aur-engineer-karein)
    * [Code Example: Mutual Information](#code-example-mutual-information)
6. [ML algorithm select karein](#ml-algorithm-select-karein)
7. [Model design aur tune karein](#model-design-aur-tune-karein)
    * [Code Example: Bias-Variance Trade-Off](#code-example-bias-variance-trade-off)
8. [Model selection ke liye cross-validation ka use kaise karein](#model-selection-ke-liye-cross-validation-ka-use-kaise-karein)
    * [Code Example: Python mein cross-validation kaise implement karein](#code-example-python-mein-cross-validation-kaise-implement-karein)
9. [Scikit-learn ke saath Parameter tuning](#scikit-learn-ke-saath-parameter-tuning)
    * [Code Example: Yellowbricks ke saath Learning aur Validation curves](#code-example-yellowbricks-ke-saath-learning-aur-validation-curves)
    * [Code Example: GridSearchCV aur pipeline ka use karke Parameter tuning](#code-example-gridsearchcv-aur-pipeline-ka-use-karke-parameter-tuning)
10. [Finance mein cross-validation ke saath Challenges](#finance-mein-cross-validation-ke-saath-challenges)
    * [Purging, embargoing, aur combinatorial CV](#purging-embargoing-aur-combinatorial-cv)


## Data se machine learning kaise kaam karti hai

ML ki kai definitions data mein meaningful patterns ke automated detection ke ird-gird ghumti hain. Do prominent examples mein shamil hain:
- AI pioneer Arthur Samuelson ne 1959 mein ML ko computer science ke ek subfield ke roop mein define kiya jo computers ko explicitly program kiye bina seekhne ki kshamta deta hai.
- Tom Mitchell, jo field ke current leaders mein se ek hain, ne 1998 mein ek well-posed learning problem ko zyada specifically pin down kiya: ek computer program kisi task aur performance measure ke sandarbh mein experience se seekhta hai ki kya task ki performance experience ke saath improve hoti hai (Mitchell, 1997).

Experience algorithm ko training data ke form mein present kiya jata hai. Problems solve karne wali machines banane ki pichli koshishon se principal difference ye hai ki algorithm decisions lene ke liye jo rules use karta hai wo humans dwara program kiye jane ke bajaye data se seekhe jate hain, jaisa ki, example ke liye, 1980s mein prominent expert systems ke case mein tha.

Algorithm aur general applications ki wide range ko cover karne wale recommended textbooks mein shamil hain:
- [An Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/), James et al (2013)
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/), Hastie, Tibshirani, and Friedman (2009)
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), Bishop (2006)
- [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html), Mitchell (1997).

### Key challenge: Given task ke liye sahi algorithm dhundhna

Automated learning ka key challenge training data mein un patterns ko pehchanna hai jo model ki learning ko naye data par generalize karte samay meaningful hon. Aise badi sankhya mein potential patterns hain jinhe model pehchan sakta hai, jabki training data phenomena ke larger set ka sirf ek sample constitute karta hai jiska algorithm future mein task perform karte samay saamna kar sakta hai.

### Supervised Learning: example se task sikhana

Supervised learning ML ka sabse commonly used type hai. Hum is book ke zyadatar chapters is category ke applications ko dedicate karenge. Term 'supervised' ek outcome variable ki maujudgi imply karta hai jo learning process ko guide karta hai—yani, ye algorithm ko task ka sahi solution sikhata hai. Supervised learning ka maksad individual samples se functional input-output relationship capture karna hai jo is relationship ko reflect karte hain aur naye data ke baare mein valid statements dekar apni learning apply karna hai.

### Unsupervised learning: Useful patterns pehchanne ke liye data explore karna

Unsupervised learning problem solve karte samay, hum sirf features observe karte hain aur outcome ka koi measurements nahi hota. Future outcomes predict karne ya variables ke beech relationships infer karne ke bajaye, unsupervised algorithms input mein wo structure pehchanne ka maksad rakhte hain jo data mein contained information ke naye representation ki anumati deta hai.

#### Trading strategies ke liye use cases: Risk management se text processing tak
Unsupervised learning ke liye kai trading use cases hain jinhe hum baad ke chapters mein cover karenge:
- Similar risk aur return characteristics wali securities ko ek saath group karna (dekhein [Chapter 13 mein hierarchical risk parity](../13_unsupervised_learning/04_hierarchical_risk_parity)
- [Principal component analysis](../13_unsupervised_learning/01_linear_dimensionality_reduction)) ya autoencoders ([Chapter 20](../20_autoencoders_for_conditional_risk_factors) ka use karke securities ki bahut badi sankhya ki performance drive karne wale risk factors ki small number dhundhna
- Documents ki body mein (example ke liye, earnings call transcripts) latent topics pehchanna jo un documents ke sabse mahatvapurn aspects comprise karte hain ([Chapter 15](../15_topic_modeling))

### Reinforcement learning: Karke seekhna, ek baar mein ek kadam

Reinforcement learning (RL) ML ka teesra type hai. Ye ek agent par center karta hai jise environment dwara provide ki gayi information ke aadhar par har time step par ek action pick karne ki zarurat hoti hai. Agent ek self-driving car ho sakta hai, board game ya video game khelne wala program ho sakta hai, ya kisi certain security market mein operate karne wali trading strategy ho sakti hai.

Introduction ke liye [Sutton aur Barto](http://www.incompleteideas.net/book/the-book-2nd.html), 2018 dekhein.

## Machine Learning Workflow

ML solution develop karne ke liye success ke chances maximize karne ke liye systematic approach ki zarurat hoti hai jabki efficiently aage badhna hota hai. Collaboration, maintenance, aur baad ke refinements ko aasan banane ke liye process ko transparent aur replicable banana bhi zaruri hai.

Process throughout iterative hai, aur alag-alag stages par effort project ke hisab se vary karega. Fir bhi, is process mein aamtaur par nimnlikhit steps shamil hone chahiye:

1. Problem frame karein, target metric identify karein, aur success define karein
2. Data source, clean, aur validate karein
3. Apne data ko samjhein aur informative features generate karein
4. Apne data ke liye suitable ek ya zyada machine learning algorithms pick karein
5. Apne models Train, test, aur tune karein
6. Original problem solve karne ke liye apne model ka use karein

### Code Example: K-nearest neighbors ke saath ML workflow

Notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb) mein kai examples hain jo house prices ke simple dataset ka use karke machine learning workflow illustrate karte hain.

- sklearn [Documentation](http://scikit-learn.org/stable/documentation.html)
- k-nearest neighbors [tutorial](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn) aur [visualization](http://vision.stanford.edu/teaching/cs231n-demos/knn/)

## Problem frame karein: goals aur metrics

Kisi bhi machine learning exercise ke liye starting point ultimate use case hota hai jise ye address karne ka maksad rakhta hai. Kabhi-kabhi, ye goal variables ke beech association ya causal relationship identify karne ke liye statistical inference hoga. Halaanki, aksar, goal trading signal yield karne ke liye outcome ka direct prediction hoga.

## Data Collect aur prepare karein

Humne market aur fundamental data ki sourcing [Chapter 2](../02_market_and_fundamental_data) mein, aur alternative data ke liye [Chapter 3](../03_alternative_data) mein address kiya tha. Hum in sources ke various examples ke saath kaam karna jari rakhenge jaise-jaise hum baad ke chapters mein various models ke application illustrate karenge.

## Features ko kaise explore, extract aur engineer karein

Individual variables ke distribution aur outcomes aur features ke beech relationships ko samajhna suitable algorithm pick karne ka aadhar hai. Ye aamtaur par visualizations jaise scatter plots se shuru hota hai, jaisa ki companion notebook mein illustrate kiya gaya hai (aur following image mein dikhaya gaya hai), lekin isme linear metrics, jaise correlation, se lekar nonlinear statistics, jaise Spearman rank correlation coefficient (jiska saamna humne information coefficient introduce karte samay kiya tha) tak numerical evaluations bhi shamil hain. Isme information-theoretic measures, jaise mutual information bhi shamil hain.

### Code Example: Mutual Information

Notebook [mutual_information](02_mutual_information.ipynb) financial data par information theory apply karta hai jo humne chapter [Alpha Factors – Research and Evaluation]((../04_alpha_factor_research) mein notebook [feature_engineering](../04_alpha_factor_research/00_data/feature_engineering.ipynb) mein create kiya tha.

## ML algorithm select karein

Is book ka shesh bhaag kai model families introduce karega, jo linear models (jo input aur output variables ke beech functional relationship ke nature ke baare mein kaafi strong assumptions banate hain) se lekar deep neural networks (jo bahut kam assumptions banate hain) tak range karte hain.

## Model design aur tune karein

ML process mein model ke generalization error ke estimates ke aadhar par model complexity diagnose aur manage karne ke steps shamil hain. Unbiased estimate ke liye ek statistically sound aur efficient procedure ki zarurat hoti hai, saath hi error metrics jo output variable type ke saath align karein, jo ye bhi determine karta hai ki hum regression, classification, ya ranking problem ke saath deal kar rahe hain.

### Code Example: Bias-Variance Trade-Off

Naye input data ke liye outcomes predict karte samay ML model jo errors karta hai use reducible aur irreducible parts mein toda ja sakta hai. Irreducible part data mein random variation (noise) ki wajah se hota hai jo measure nahi kiya gaya hai, jaise relevant lekin missing variables ya natural variation.

Notebook [bias_variance](03_bias_variance.ipynb) increasingly complex polynomials ka use karke cosine function approximate karke aur in-sample error measure karke overfitting demonstrate karta hai. Ye varying complexity ke polynomial seekhne ke liye kuch added noise (n = 30) ke saath 10 random samples draw karta hai. Har baar, model naye data points predict karta hai aur hum in predictions ke liye mean-squared error capture karte hain, saath hi in errors ka standard deviation bhi. Ye kuch added noise ke saath ninth degree ke cosine function ka Taylor series approximation seekhne ki koshish karke overfitting versus underfitting ke impact ko illustrate karne ke liye aage badhta hai. Following diagram mein, hum true function ke random samples draw karte hain aur polynomials fit karte hain jo underfit, overfit karte hain, aur flexibility ki approximately correct degree provide karte hain.

## Model selection ke liye cross-validation ka use kaise karein

Jab aapke use case ke liye kai candidate models (yani, algorithms) available hote hain, to unme se ek ko chunne ke act ko model selection problem kaha jata hai. Model selection ka maksad us model ko identify karna hai jo naye data ko dekhte huye lowest prediction error produce karega.

### Code Example: Python mein cross-validation kaise implement karein

Script [cross_validation](04_cross_validation.py) ye dikhakar ki ten observations wale mock dataset ke indices train aur test set mein kaise assign kiye jate hain, data ko training aur test sets mein split karne ke various options illustrate karta hai.
 
## Scikit-learn ke saath Parameter tuning

Model selection mein aamtaur par alag-alag algorithms (jaise linear regression aur random forest) ya alag-alag configurations ka use karke models ki out-of-sample performance ka repeated cross-validation shamil hota hai. Alag-alag configurations mein hyperparameters mein changes ya alag-alag variables ka inclusion ya exclusion shamil ho sakta hai.

### Code Example: Yellowbricks ke saath Learning aur Validation curves

Notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb)) various model selection techniques ke use ko illustrate karte huye learning aur validation curves ka use demonstrate karta hai.

- Yellowbrick: Machine Learning Visualization [docs](http://www.scikit-yb.org/en/latest/)

### Code Example: GridSearchCV aur pipeline ka use karke Parameter tuning

Kyunki hyperparameter tuning machine learning workflow ka ek key ingredient hai, isliye is process ko automate karne ke liye tools hain. sklearn library mein ek GridSearchCV interface shamil hai jo parallel mein parameters ke sabhi combinations ko cross-validate karta hai, result capture karta hai, aur automatically full dataset par cross-validation ke dauran best perform karne wale parameter setting ka use karke model train karta hai.

Practice mein, training aur validation sets ko aksar cross-validation se pehle kuch processing ki zarurat hoti hai. Scikit-learn Pipeline offer karta hai taaki GridSearchCV dwara facilitated automated hyperparameter tuning mein kisi bhi requisite feature-processing steps ko bhi automate kiya ja sake.

In tools ko action mein dekhne ke liye included machine_learning_workflow.ipynb notebook mein implementation examples dekhein.

Notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb)) bhi in tools ka use demonstrate karta hai.

## Finance mein cross-validation ke saath Challenges

Ab tak charcha kiye gaye cross-validation methods ke liye ek key assumption training ke liye available samples ka independent and identical (iid) distribution hai.
Financial data ke liye, aksar aisa nahi hota hai. Iske vipreet, financial data na to independently aur na hi identically distributed hota hai kyunki serial correlation aur time-varying standard deviation hota hai, jise heteroskedasticity bhi kaha jata hai.

### Purging, embargoing, aur combinatorial CV

Financial data ke liye, labels aksar overlapping data points se derive kiye jate hain kyunki returns multiple periods mein prices se compute kiye jate hain. Trading strategies ke sandarbh mein, model ke prediction ke results, jo asset mein position lena imply kar sakte hain, sirf baad mein pata chal sakte hain, jab ye decision evaluate kiya jata hai—example ke liye, jab position close out ki jati hai.

Resulting risks mein test se training set mein information ka leaking shamil hai, jisse performance artificially inflated hone ki sambhavna hoti hai. Ise resolve karne ke liye ye ensure kiya jana chahiye ki saara data point-in-time ho—yani, vastav mein available ho aur us time par known ho jab ise model ke liye input ke roop mein use kiya ja raha ho. Marcos Lopez de Prado ne [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) mein cross-validation ke liye financial data ke in challenges ko address karne ke liye kai methods propose kiye hain:

- Purging: Un training data points ko eliminate karein jahan evaluation validation set mein point-in-time data point ke prediction ke baad hoti hai taaki look-ahead bias se bacha ja sake.
- Embargoing: Un training samples ko aur eliminate karein jo test period ko follow karte hain.
