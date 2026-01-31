# Financial Feature Engineering: Alpha Factors ko research kaise karein

Algorithmic trading strategies un signals se chalti hain jo batate hain ki kisi benchmark (jaise index) ke mukable behtar returns paane ke liye assets ko kab khareedna (buy) ya bechna (sell) hai. Asset ke return ka wo hissa jo benchmark ke exposure se explain nahi hota, use alpha kaha jata hai, aur isliye wo signals jo aise uncorrelated returns paida karne ka maksad rakhte hain, unhe alpha factors bhi kaha jata hai.

Agar aap ML se pehle se waqif hain, to aap jaante honge ki successful predictions ke liye feature engineering ek mukhya hissa (key ingredient) hai. Trading mein bhi aisa hi hai. Halaanki, investment mein dashakon (decades) ki research shamil hai ki markets kaise kaam karte hain aur natijan price movements ko explain ya predict karne ke liye kaun se features dusron se behtar kaam kar sakte hain. Ye chapter aapke apne alpha factors ki khoj ke liye ek shuruwati bindu (starting point) ke roop mein ek overview deta hai.

Ye chapter un mukhya tools ko bhi prastut karta hai jo alpha factors ko compute aur test karna aasan banate hain. Hum highlight karenge ki kaise NumPy, pandas aur TA-Lib libraries data manipulation ko aasan banati hain aur wavelets aur Kalman filter jaise popular smoothing techniques prastut karenge jo data mein noise kam karne mein madad karte hain.

Hum ye bhi preview karenge ki (traditional) alpha factors ki predictive performance evaluate karne ke liye aap trading simulator Zipline ka use kaise kar sakte hain. Hum information coefficient aur factor turnover jaise mukhya alpha factor metrics par charcha karenge. Machine learning ka use karne wali backtesting trading strategies ka gehra parichay [Chapter 6](../08_ml4t_workflow) mein diya gaya hai, jo **ML4T workflow** ko cover karta hai jiska use hum puri kitaab mein trading strategies evaluate karne ke liye karenge.

Is topic par additional material ke liye kripya [Appendix - Alpha Factor Library](../24_alpha_factor_library) dekhein, jisme dher saare code examples shamil hain jo alpha factors ki ek broad range compute karte hain.

## Vishay Soochi (Content)

1. [Alpha Factors in practice: data se signals tak](#alpha-factors-in-practice-data-se-signals-tak)
2. [Dashakon ki Factor Research par nirmaan](#dashakon-ki-factor-research-par-nirmaan)
    * [References](#references)
3. [Engineering alpha factors jo returns predict karte hain](#engineering-alpha-factors-jo-returns-predict-karte-hain)
    * [Code Example: pandas aur NumPy ka use karke factors engineer kaise karein](#code-example-pandas-aur-numpy-ka-use-karke-factors-engineer-kaise-karein)
    * [Code Example: Technical alpha factors banane ke liye TA-Lib ka use kaise karein](#code-example-technical-alpha-factors-banane-ke-liye-ta-lib-ka-use-kaise-karein)
    * [Code Example: Kalman Filter ke saath apne Alpha Factors ko denoise kaise karein](#code-example-kalman-filter-ke-saath-apne-alpha-factors-ko-denoise-kaise-karein)
    * [Code Example: Wavelets ka use karke apne noisy signals ko preprocess kaise karein](#code-example-wavelets-ka-use-karke-apne-noisy-signals-ko-preprocess-kaise-karein)
    * [Resources](#resources)
4. [Signals se trades tak: `Zipline` ke saath backtesting](#signals-se-trades-tak-zipline-ke-saath-backtesting)
    * [Code Example: Single-factor strategy backtest karne ke liye Zipline ka use kaise karein](#code-example-single-factor-strategy-backtest-karne-ke-liye-zipline-ka-use-kaise-karein)
    * [Code Example: Quantopian platform par diverse data sources se factors combine karna](#code-example-quantopian-platform-par-diverse-data-sources-se-factors-combine-karna)
    * [Code Example: Signal aur noise alag karna – alphalens ka use kaise karein](#code-example-signal-aur-noise-alag-karna-alphalens-ka-use-kaise-karein)
5. [Vaikalpik (Alternative) Algorithmic Trading Libraries aur Platforms](#vaikalpik-algorithmic-trading-libraries-aur-platforms)

## Alpha Factors in practice: data se signals tak

Alpha factors market, fundamental, aur alternative data ke transformations hain jinme predictive signals hote hain. Inhe un risks ko pakadne ke liye design kiya jata hai jo asset returns ko drive karte hain. Factors ka ek set fundamental, economy-wide variables ko describe karta hai jaise growth, inflation, volatility, productivity, aur demographic risk. Dusra set tradeable investment styles se bana hota hai jaise market portfolio, value-growth investing, aur momentum investing.

Aise factors bhi hain jo financial markets ki economics ya institutional setting, ya investor behavior (isme is behavior ke known biases bhi shamil hain) ke aadhar par price movements ko explain karte hain. Factors ke piche ki economic theory rational ho sakti hai, jahan factors ke long run mein high returns hote hain taaki bure waqt mein unke low returns ki bharpayi ho sake, ya behavioral ho sakti hai, jahan factor risk premiums agents ke possibly biased, ya puri tarah rational na hone wale behavior se aate hain jise arbitrage se hataya nahi jata.

## Dashakon ki Factor Research par nirmaan

Ek adarsh (idealized) duniya mein, risk factors ki categories ek dusre se independent (orthogonal) honi chahiye, positive risk premia deni chahiye, aur ek complete set banana chahiye jo risk ke sabhi dimensions ko span kare aur kisi di gayi class mein assets ke liye systematic risks ko explain kare. Practice mein, ye requirements sirf lagbhag (approximately) hi hold karti hain.

### References

- [Dissecting Anomalies](http://schwert.ssb.rochester.edu/f532/ff_JF08.pdf) by Eugene Fama and Ken French (2008)
- [Explaining Stock Returns: A Literature Review](https://www.ifa.com/pdfs/explainingstockreturns.pdf) by James L. Davis (2001)
- [Market Efficiency, Long-Term Returns, and Behavioral Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=15108) by Eugene Fama (1997)
- [The Efficient Market Hypothesis and It's Critics](https://pubs.aeaweb.org/doi/pdf/10.1257/089533003321164958) by Burton Malkiel (2003)
- [The New Palgrave Dictionary of Economics](https://www.palgrave.com/us/book/9780333786765) (2008) by Steven Durlauf and Lawrence Blume, 2nd ed.
- [Anomalies and Market Efficiency](https://www.nber.org/papers/w9277.pdf) by G. William Schwert25 (Ch. 15 in Handbook of the- "Economics of Finance", by Constantinides, Harris, and Stulz, 2003)
- [Investor Psychology and Asset Pricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=265132), by David Hirshleifer (2001)
- [Practical advice for analysis of large, complex data sets](https://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html), Patrick Riley, Unofficial Google Data Science Blog

## Engineering alpha factors jo returns predict karte hain

Key factor categories, unke rationale aur popular metrics ki conceptual understanding ke aadhar par, ek mukhya kaam naye factors ki pehchan karna hai jo pehle bataye gaye return drivers dwara embodied risks ko behtar dhang se pakad sakein, ya naye factors dhundhna hai. Kisi bhi case mein, innovative factors ki performance ko known factors se compare karna zaruri hoga taaki incremental signal gains ki pehchan ki ja sake.

### Code Example: pandas aur NumPy ka use karke factors engineer kaise karein

[data](00_data) directory mein notebook [feature_engineering.ipynb](00_data/feature_engineering.ipynb) (iska Hindi version `feature_engineering_HINDI.ipynb` bhi hai) dikhata hai ki basic factors ko engineer kaise karein.

### Code Example: Technical alpha factors banane ke liye TA-Lib ka use kaise karein

Notebook [how_to_use_talib](02_how_to_use_talib.ipynb) (iska Hindi version `how_to_use_talib_HINDI.ipynb` bhi hai) TA-Lib ke upyog ko darshata hai, jisme common technical indicators ki ek broad range shamil hai. In indicators mein ye baat common hai ki ye sirf market data use karte hain, yani price aur volume information.

**Appendix** mein notebook [common_alpha_factors](../24_alpha_factor_library/02_common_alpha_factors.ipynb) mein darjano additional examples hain.

### Code Example: Kalman Filter ke saath apne Alpha Factors ko denoise kaise karein

Notebook [kalman_filter_and_wavelets](03_kalman_filter_and_wavelets.ipynb) (iska Hindi version `kalman_filter_and_wavelets_HINDI.ipynb` bhi hai) smoothing ke liye `PyKalman` package ka use karke Kalman filter ka upyog dikhata hai; hum [Chapter 9](../09_time_series_models) mein bhi iska use karenge jab hum ek pairs trading strategy develop karenge.

### Code Example: Wavelets ka use karke apne noisy signals ko preprocess kaise karein

Notebook [kalman_filter_and_wavelets](03_kalman_filter_and_wavelets.ipynb) ye bhi dikhata hai ki `PyWavelets` package ka use karke wavelets ke saath kaise kaam karein.

### Resources

- [Fama French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) Data Library
- [numpy](https://numpy.org/) website
    - [Quickstart Tutorial](https://numpy.org/devdocs/user/quickstart.html)
- [pandas](https://pandas.pydata.org/) website
    - [User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
    - [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
    - [Python Pandas Tutorial: A Complete Introduction for Beginners](https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/)
- [alphatools](https://github.com/marketneutral/alphatools) - Quantitative finance research tools in Python
- [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) - Package based on the work of Dr Marcos Lopez de Prado regarding his research with respect to Advances in Financial Machine Learning
- [PyKalman](https://pykalman.github.io/) documentation
- [Tutorial: The Kalman Filter](http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf)
- [Understanding and Applying Kalman Filtering](http://biorobotics.ri.cmu.edu/papers/sbp_papers/integrated3/kleeman_kalman_basics.pdf)
- [How a Kalman filter works, in pictures](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) - Wavelet Transforms in Python
- [An Introduction to Wavelets](https://www.eecis.udel.edu/~amer/CISC651/IEEEwavelet.pdf) 
- [The Wavelet Tutorial](http://web.iitd.ac.in/~sumeet/WaveletTutorial.pdf)
- [Wavelets for Kids](http://www.gtwavelet.bme.gatech.edu/wp/kidsA.pdf)
- [The Barra Equity Risk Model Handbook](https://www.alacra.com/alacra/help/barra_handbook_GEM.pdf)
- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [Modern Investment Management: An Equilibrium Approach](https://www.amazon.com/Modern-Investment-Management-Equilibrium-Approach/dp/0471124109) by Bob Litterman, 2003
- [Quantitative Equity Portfolio Management: Modern Techniques and Applications](https://www.crcpress.com/Quantitative-Equity-Portfolio-Management-Modern-Techniques-and-Applications/Qian-Hua-Sorensen/p/book/9781584885580) by Edward Qian, Ronald Hua, and Eric Sorensen
- [Spearman Rank Correlation](https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php)

## Signals se trades tak: `Zipline` ke saath backtesting

Open source [zipline](https://zipline.ml4trading.io/index.html) library ek event-driven backtesting system hai jise crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) dwara maintain aur production mein use kiya jata hai taaki algorithm-development aur live-trading aasan ho sake. Ye trade events par algorithm ke reaction ko automate karta hai aur ise current aur historical point-in-time data deta hai jo look-ahead bias se bachaata hai.

- [Chapter 8](../08_ml4t_workflow) mein Zipline ka zyada vistrit (comprehensive) parichay diya gaya hai.
- Kripya `installation` folder mein [instructions](../installation) follow karein, jisme **known issues** ko address karna bhi shamil hai.

### Code Example: Single-factor strategy backtest karne ke liye Zipline ka use kaise karein

Notebook [single_factor_zipline](04_single_factor_zipline.ipynb) (iska Hindi version `single_factor_zipline_HINDI.ipynb` bhi hai) ek simple mean-reversion factor develop aur test karta hai jo ye measure karta hai ki recent performance historical average se kitna deviate (bhatakna) hua hai. Short-term reversal ek common strategy hai jo is weakly predictive pattern ka fayda uthati hai ki stock price increases ke horizons (ek minute se kam se lekar ek mahine tak) par wapas niche mean-revert hone ki sambhavna hoti hai.

### Code Example: Quantopian platform par diverse data sources se factors combine karna

Quantopian research environment predictive alpha factors ki rapid testing ke liye tailored hai. Process bahut similar hai kyunki ye `zipline` par banta hai, lekin data sources tak kahin zyada rich access deta hai.

Notebook [multiple_factors_quantopian_research](05_multiple_factors_quantopian_research.ipynb) dikhata hai ki alpha factors ko na sirf market data se (jaisa pehle kiya gaya tha) balki fundamental aur alternative data se bhi kaise compute karein.

### Code Example: Signal aur noise alag karna – alphalens ka use kaise karein

Notebook [performance_eval_alphalens](06_performance_eval_alphalens.ipynb) (iska Hindi version `performance_eval_alphalens_HINDI.ipynb` bhi hai) predictive (alpha) factors ki performance analysis ke liye [alphalens](http://quantopian.github.io/alphalens/) library introduce karta hai, jise Quantopian ne open-source kiya hai. Ye dikhata hai ki ye backtesting library `zipline` aur portfolio performance aur risk analysis library `pyfolio` (jise hum agle chapter mein explore karenge) ke saath kaise integrate hota hai.

`alphalens` alpha factors ki predictive power ke analysis ko aasan banata hai:
- Signals ka subsequent (baad ke) returns ke saath correlation
- Signals (ke subset) par based equal ya factor-weighted portfolio ki profitability
- Potential trading costs indicate karne ke liye factors ka turnover
- Specific events ke dauran factor-performance
- Sector ke hisab se upar walon ka breakdown

Analysis `tearsheets` ya individual computations aur plots ka use karke kiya ja sakta hai. Space bachaane ke liye tearsheets ko online repo mein dikhaya gaya hai.

- [Yahan](https://github.com/quantopian/alphalens/blob/master/alphalens/examples/alphalens_tutorial_on_quantopian.ipynb) dekhein Quantopian dwara ek detailed `alphalens` tutorial ke liye.

## Vaikalpik (Alternative) Algorithmic Trading Libraries aur Platforms

- [QuantConnect](https://www.quantconnect.com/)
- [Alpha Trading Labs](https://www.alphalabshft.com/)
    - Alpha Trading Labs ab active nahi hai
- [WorldQuant](https://www.worldquantvrc.com/en/cms/wqc/home/)
- Python Algorithmic Trading Library [PyAlgoTrade](http://gbeced.github.io/pyalgotrade/)
- [pybacktest](https://github.com/ematvey/pybacktest)
- [Trading with Python](http://www.tradingwithpython.com/)
- [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=5041)
