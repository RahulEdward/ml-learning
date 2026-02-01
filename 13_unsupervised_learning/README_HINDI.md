# Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity 

Unsupervised learning hai useful when a dataset contain karta hai only features aur no measurement ka the outcome, or when hum want to extract information independent from the outcome. Instead ka predicting future outcomes, the goal hai to learn an informative representation ka the data that hai useful ke liye solving another task, including the exploration ka a data set. Examples include the identification ka topics to summarize documents (see [Chapter 14](../14_topic_modeling), the reduction ka the number ka features to lower the risk ka overfitting aur computational cost ke liye supervised learning, or to group similar observations as illustrated by the use ka clustering ke liye asset allocation at the end ka this chapter.

Dimensionality reduction aur clustering hain the main tasks ke liye unsupervised learning: 
- Dimensionality reduction transforms the existing features into a new, smaller set while minimizing the loss ka information. A broad range ka algorithms exists that differ by how they measure the loss ka information, whether they apply linear or non-linear transformations or the constraints they impose on the new feature set. 
- Clustering algorithms identify aur group similar observations or features instead ka identifying new features. Algorithms differ mein how they define the similarity ka observations aur their assumptions about the resulting groups.

More specifically, this chapter covers:
- How principal aur independent component analysis (PCA aur ICA) perform linear dimensionality reduction
- Identifying data-driven risk factors aur eigenportfolios from asset returns use karke PCA
- Effectively visualizing nonlinear, high-dimensional data use karke manifold learning
- use karke T-SNE aur UMAP to explore high-dimensional image data
- How k-means, hierarchical, aur density-based clustering algorithms work
- use karke agglomerative clustering to build robust portfolios ke saath hierarchical risk parity

## Vishay-suchi (Content)

1. [Code Example: the curse ka dimensionality](#code-example-the-curse-ka-dimensionality)
2. [Linear Dimensionality Reduction](#linear-dimensionality-reduction)
    * [Code Example: Principal Component Analysis](#code-example-principal-component-analysis)
        - [Visualizing key ideas behind PCA ](#visualizing-key-ideas-behind-pca-)
        - [How the PCA algorithm works](#how-the-pca-algorithm-works)
    * [References](#references)
3. [Code Examples: PCA ke liye Trading ](#code-examples-pca-ke liye-trading-)
    * [Data-driven risk factors](#data-driven-risk-factors)
    * [Eigenportfolios](#eigenportfolios)
    * [References](#references-2)
4. [Independent Component Analysis](#independent-component-analysis)
5. [Manifold Learning](#manifold-learning)
    * [Code Example: what a manifold looks like ](#code-example-what-a-manifold-looks-like-)
    * [Code Example: Local Linear Embedding](#code-example-local-linear-embedding)
    * [References](#references-3)
6. [Code Examples: visualizing high-dimensional image aur asset price data ke saath manifold learning](#code-examples-visualizing-high-dimensional-image-aur-asset-price-data-ke saath-manifold-learning)
    * [t-distributed stochastic neighbor embedding (t-SNE)](#t-distributed-stochastic-neighbor-embedding-t-sne)
    * [UMAP](#umap)
7. [Cluster Algorithms](#cluster-algorithms)
    * [Code example: comparing cluster algorithms](#code-example-comparing-cluster-algorithms)
    * [Code example: k-Means](#code-example-k-means)
        - [The algorithm](#the-algorithm)
        - [Evaluating the results](#evaluating-the-results)
    * [Code example: Hierarchical Clustering](#code-example-hierarchical-clustering)
    * [Code example: Density-Based Clustering](#code-example-density-based-clustering)
    * [Code example: Gaussian Mixture Models](#code-example-gaussian-mixture-models)
    * [Code example: Hierarchical Risk Parity](#code-example-hierarchical-risk-parity)
        - [The algorithm](#the-algorithm-2)
        - [Backtest comparison with alternatives](#backtest-comparison-with-alternatives)
    * [References](#references-4)

## Code Example: the curse ka dimensionality

The number ka dimensions ka a dataset matter because each new dimension can add signal concerning an outcome. However, there hai also a downside known as the curse ka dimensionality: as the number ka independent features grows while the number ka observations remains constant, the average distance between data points also grows, aur the density ka the feature space drops exponentially. The implications ke liye machine learning hain dramatic because prediction becomes much harder when observations hain more distant, i.e., different from each other.

Notebook [curse_of_dimensionality](01_linear_dimensionality_reduction/00_curse_of_dimensionality.ipynb) simulates how the average aur minimum distances between data points increase as the number ka dimensions grows.

## Linear Dimensionality Reduction

Linear dimensionality reduction algorithms compute linear combinations that translate, rotate, aur rescale the original features to capture significant variation mein the data, subject to constraints on the characteristics ka the new features.

Yeh section introduces these two algorithms aur then illustrates how to apply PCA to asset returns to learn risk factors from the data, aur to build so-called eigen portfolios ke liye systematic trading strategies.

- [Dimension Reduction: A Guided Tour](https://www.microsoft.com/en-us/research/publication/dimension-reduction-a-guided-tour-2/), Chris J.C. Burges, Foundations aur Trends mein Machine Learning, January 2010

### Code Example: Principal Component Analysis

PCA finds principal components as linear combinations ka the existing features aur use karta hai these components to represent the original data. The number ka components hai a hyperparameter that determines the target dimensionality aur needs to be equal to or smaller than the number ka observations or columns, whichever hai smaller.

#### Visualizing key ideas behind PCA 

Notebook [pca_key_ideas](01_linear_dimensionality_reduction/01_pca_key_ideas.ipynb) visualizes principal components mein 2D aur 3D.

PCA aims to capture most ka the variance mein the data, to make it easy to recover the original features, aur that each component adds information. It reduces dimensionality by projecting the original data into the principal component space. PCA makes several assumptions that hain important to keep mein mind. These include:
- high variance implies a high signal-to-noise ratio
- the data hai standardized so that the variance hai comparable across features
- linear transformations capture the relevant aspects ka the data, aur
- higher-order statistics beyond the first aur second moment do not matter, which implies that the data has a normal distribution

The emphasis on the first aur second moments align ke saath standard risk/return metrics, but the normality assumption may conflict ke saath the characteristics ka market data.

#### How the PCA algorithm works

Notebook [the_math_behind_pca](01_linear_dimensionality_reduction/02_the_math_behind_pca.ipynb) illustrate the computation ka principal components.

### References

- [Mixtures ka Probabilistic Principal Component Analysers](http://www.miketipping.com/papers/met-mppca.pdf), Michael E. Tipping aur Christopher M. Bishop, Neural Computation 11(2), pp 443–482. MIT Press
- [Finding Structure ke saath Randomness: Probabilistic Algorithms ke liye Constructing Approximate Matrix Decompositions](http://users.cms.caltech.edu/~jtropp/papers/HMT11-Finding-Structure-SIREV.pdf), N. Halko†, P. G. Martinsson, J. A. Tropp, SIAM REVIEW, Vol. 53, No. 2, pp. 217–288
- [Relationship between SVD aur PCA. How to use SVD to perform PCA?](https://stats.stackexchange.com/questions/134282/relationship-between-svd-aur-pca-how-to-use-svd-to-perform-pca), excellent technical CrossValidated StackExchange answer ke saath visualization

## Code Examples: PCA ke liye Trading 

PCA hai useful ke liye algorithmic trading mein several respects, including the data-driven derivation ka risk factors by applying PCA to asset returns, aur the construction ka uncorrelated portfolios based on the principal components ka the correlation matrix ka asset returns.
 
### Data-driven risk factors

mein [Chapter 07 - Linear Models](../07_linear_models/02_fama_macbeth.ipynb), hum explored risk factor models used mein quantitative finance to capture the main drivers ka returns. These models explain differences mein returns on assets based on their exposure to systematic risk factors aur the rewards associated ke saath these factors.
 
mein particular, hum explored the Fama-French approach that specifies factors based on prior knowledge about the empirical behavior ka average returns, treats these factors as observable, aur then estimates risk model coefficients use karke linear regression. An alternative approach treats risk factors as latent variables aur use karta hai factor analytic techniques like PCA to simultaneously estimate the factors aur how the drive returns from historical returns.

- Notebook [pca_and_risk_factor_models](01_linear_dimensionality_reduction/03_pca_and_risk_factor_models.ipynb) demonstrate karta hai how this method derives factors mein a purely statistical or data-driven way ke saath the advantage ka not requiring ex-ante knowledge ka the behavior ka asset returns.
 
### Eigenportfolios

Another application ka PCA involves the covariance matrix ka the normalized returns. The principal components ka the correlation matrix capture most ka the covariation among assets mein descending order aur hain mutually uncorrelated. Moreover, hum can use standardized principal components as portfolio weights. 

Notebook [pca_and_eigen_portfolios](01_linear_dimensionality_reduction/04_pca_and_eigen_portfolios.ipynb) illustrates how to create Eigenportfolios.

### References

- [Characteristics hain Covariances: A Unified Model ka Risk aur Return](http://www.nber.org/2018LTAM/kelly.pdf), Kelly, Pruitt aur Su, NBER, 2018
- [Statistical Arbitrage mein the U.S. Equities Market](https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf), Marco Avellaneda aur Jeong-Hyun Lee, 2008

## Independent Component Analysis

Independent component analysis (ICA) hai another linear algorithm that identifies a new basis to represent the original data but pursues a different objective than PCA. See [Hyvärinen aur Oja](https://www.sciencedirect.com/science/article/pii/S0893608000000265) (2000) ke liye a detailed introduction.
 
ICA emerged mein signal processing, aur the problem it aims to solve hai called blind source separation. It hai typically framed as the cocktail party problem where a given number ka guests hain speaking at the same time so that a single microphone would record overlapping signals. ICA assumes there hain as many different microphones as there hain speakers, each placed at different locations so that it records a different mix ka the signals. ICA then aims to recover the individual signals from the different recordings.

- [Independent Component Analysis: Algorithms aur Applications](https://www.sciencedirect.com/science/article/pii/S0893608000000265), Aapo Hyvärinen aur Erkki Oja, Neural Networks, 2000
- [Independent Components Analysis](http://cs229.stanford.edu/notes/cs229-notes11.pdf), CS229 Lecture Notes, Andrew Ng
- [Common factors mein prices, order flows, aur liquidity](https://www.sciencedirect.com/science/article/pii/S0304405X0000091X), Hasbrouck aur Seppi, Journal ka Financial Economics, 2001
- [Volatility Modelling ka Multivariate Financial Time Series by use karke ICA-GARCH Models](https://link.springer.com/chapter/10.1007/11508069_74), Edmond H. C. Wu, Philip L. H. Yu, mein: Gallagher M., Hogan J.P., Maire F. (eds) Intelligent Data Engineering aur Automated Learning - IDEAL 2005
- [The Prediction Performance ka Independent Factor Models](http://www.cs.cuhk.hk/~lwchan/papers/icapred.pdf), Chan, mein: proceedings ka the 2002 IEEE International Joint Conference on Neural Networks
- [An Overview ka Independent Component Analysis aur Its Applications](http://www.informatica.si/ojs-2.4.3/index.php/informatica/article/download/334/333), Ganesh R. Naik, Dinesh K Kumar, Informatica 2011

## Manifold Learning

The manifold hypothesis emphasizes that high-dimensional data often lies on or near a lower-dimensional non-linear manifold that hai embedded mein the higher dimensional space. 

[Manifold learning](https://scikit-learn.org/stable/modules/manifold.html) aims to find the manifold ka intrinsic dimensionality aur then represent the data mein this subspace. A simplified example use karta hai a road as one-dimensional manifolds mein a three-dimensional space aur identifies data points use karke house numbers as local coordinates.

### Code Example: what a manifold looks like 

Notebook [manifold_learning_intro](02_manifold_learning/01_manifold_learning_intro.ipynb) contain karta hai several examples, including the two-dimensional swiss roll that illustrates the topological structure ka manifolds. 

### Code Example: Local Linear Embedding

Several techniques approximate a lower dimensional manifold. One example hai [locally-linear embedding](https://cs.nyu.edu/~roweis/lle/) (LLE) that was developed mein 2000 by Sam Roweis aur Lawrence Saul.
 
- Notebook [manifold_learning_lle](02_manifold_learning/02_manifold_learning_lle.ipynb) demonstrate karta hai how it ‘unrolls’ the swiss roll. ke liye each data point, LLE identifies a given number ka nearest neighbors aur computes weights that represent each point as a linear combination ka its neighbors. It finds a lower-dimensional embedding by linearly projecting each neighborhood on global internal coordinates on the lower-dimensional manifold aur can be thought ka as a sequence ka PCA applications.

The generic examples use the following datasets:

- [MNIST Data](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)

### References

- [Locally Linear Embedding](https://cs.nyu.edu/~roweis/lle/), Sam T. Roweis aur Lawrence K. Saul (LLE author website)

## Code Examples: visualizing high-dimensional image aur asset price data ke saath manifold learning

### t-distributed stochastic neighbor embedding (t-SNE)

[t-SNE](https://lvdmaaten.github.io/tsne/) hai an award-winning algorithm developed mein 2010 by Laurens van der Maaten aur Geoff Hinton to detect patterns mein high-dimensional data. It takes a probabilistic, non-linear approach to locating data on several different, but related low-dimensional manifolds. The algorithm emphasizes keeping similar points together mein low dimensions, as opposed to maintaining the distance between points that hain apart mein high dimensions, which results from algorithms like PCA that minimize squared distances. 

- [Visualizing Data use karke t-SNE](http://www.cs.toronto.edu/~hinton/absps/tsne.pdf), van der Maaten, Hinton, Journal ka Machine Learning Research, 2008
- [Visualizing Time-Dependent Data use karke Dynamic t-SNE](http://www.cs.rug.nl/~alext/PAPERS/EuroVis16/paper.pdf), Rauber, Falcão, Telea, Eurographics Conference on Visualization (EuroVis) 2016
- [t-Distributed Stochastic Neighbor Embedding Wins Merck Viz Challenge](http://blog.kaggle.com/2012/11/02/t-distributed-stochastic-neighbor-embedding-wins-merck-viz-challenge/), Kaggle Blog 2016
- [t-SNE: Google Tech Talk](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw), van der Maaten, 2013
- [Parametric t-SNE](https://github.com/kylemcdonald/Parametric-t-SNE), fast t-SNE implementation use karke Keras by Kyle McDonald

### UMAP

[UMAP](https://github.com/lmcinnes/umap)) hai a more recent algorithm ke liye visualization aur general dimensionality reduction. It assumes the data hai uniformly distributed on a locally connected manifold aur looks ke liye the closest low-dimensional equivalent use karke fuzzy topology. It use karta hai a neighbors parameter that impacts the result similarly as perplexity above.

It hai faster aur hence scales better to large datasets than t-SNE, aur sometimes preserves global structure than better than t-SNE. It can also work ke saath different distance functions, including, e.g., cosine similarity that hai used to measure the distance between word count vectors.

- [UMAP: Uniform Manifold Approximation aur Projection ke liye Dimension Reduction](https://arxiv.org/abs/1802.03426), Leland McInnes, John Healy, 2018

- Notebooks [manifold_learning_tsne_umap](02_manifold_learning/03_manifold_learning_tsne_umap.ipynb) aur [manifold_learning_asset_prices](02_manifold_learning/04_manifold_learning_asset_prices.ipynb) demonstrate the usage ka both t-SNE aur UMAP ke saath various data sets, including equity returns.

## Cluster Algorithms

Both clustering aur dimensionality reduction summarize the data. Dimensionality reduction compresses the data by representing it use karke new, fewer features that capture the most relevant information. Clustering algorithms, mein contrast, assign existing observations to subgroups that consist ka similar data points.

Clustering can serve to better understand the data through the lens ka categories learned from continuous variables. It also permits automatically categorizing new objects according to the learned criteria. Examples ka related applications include hierarchical taxonomies, medical diagnostics, or customer segmentation. Alternatively, clusters can be used to represent groups as prototypes, use karke e.g. the midpoint ka a cluster as the best representatives ka learned grouping. An example application includes image compression.

Clustering algorithms differ ke saath respect to their strategy ka identifying groupings:
- Combinatorial algorithms select the most coherent ka different groupings ka observations
- Probabilistic modeling estimates distributions that most likely generated the clusters
- Hierarchical clustering finds a sequence ka nested clusters that optimizes coherence at any given stage

Algorithms also differ by the notion ka what constitutes a useful collection ka objects that needs to match the data characteristics, domain aur the goal ka the applications. Types ka groupings include:
- Clearly separated groups ka various shapes
- Prototype- or center-based, compact clusters
- Density-based clusters ka arbitrary shape
- Connectivity- or graph-based clusters

Important additional aspects ka a clustering algorithm include whether 
- it requires exclusive cluster membership, 
- makes hard, i.e., binary, or soft, probabilistic assignment, aur 
- hai complete aur assigns all data points to clusters.

### Code example: comparing cluster algorithms

Notebook [clustering_algos](03_clustering_algorithms/01_clustering_algos.ipynb) compares the clustering results ke liye several algorithm use karke toy dataset designed to test clustering algorithms.

### Code example: k-Means

k-Means hai the most well-known clustering algorithm aur was first proposed by Stuart Lloyd at Bell Labs mein 1957. 

#### The algorithm

The algorithm finds K centroids aur assigns each data point to exactly one cluster ke saath the goal ka minimizing the within-cluster variance (called inertia). It typically use karta hai Euclidean distance but other metrics can also be used. k-Means assumes that clusters hain spherical aur ka equal size aur ignores the covariance among features.

- Notebook [kmeans_implementation](03_clustering_algorithms/02_kmeans_implementation.ipynb) demonstrate karta hai how the k-Means algorithm works.

#### Evaluating the results

Cluster quality metrics help select among alternative clustering results. 

- Notebook [kmeans_evaluation ](03_clustering_algorithms/03_kmeans_evaluation.ipynb) illustrates how to evaluate clustering quality use karke inertia aur the [silhouette score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

### Code example: Hierarchical Clustering

Hierarchical clustering avoids the need to specify a target number ka clusters because it assumes that data can successively be merged into increasingly dissimilar clusters. It does not pursue a global objective but decides incrementally how to produce a sequence ka nested clusters that range from a single cluster to clusters consisting ka the individual data points.

While hierarchical clustering does not have hyperparameters like k-Means, the measure ka dissimilarity between clusters (as opposed to individual data points) has an important impact on the clustering result. The options differ as follows:

- Single-link: distance between nearest neighbors ka two clusters
- Complete link: maximum distance between respective cluster members
- Group average
- Ward’s method: minimize within-cluster variance

Notebook [hierarchical_clustering](03_clustering_algorithms/04_hierarchical_clustering.ipynb) demonstrate karta hai how this algorithm works, aur how to visualize aur evaluate the results.  

### Code example: Density-Based Clustering

Density-based clustering algorithms assign cluster membership based on proximity to other cluster members. They pursue the goal ka identifying dense regions ka arbitrary shapes aur sizes. They do not require the specification ka a certain number ka clusters but instead rely on parameters that define the size ka a neighborhood aur a density threshold.

Notebook [density_based_clustering](03_clustering_algorithms/05_density_based_clustering.ipynb) demonstrate karta hai how DBSCAN aur hierarchical DBSCAN work.

- [Pairs Trading ke saath density-based clustering aur cointegration](https://www.quantopian.com/posts/pairs-trading-ke saath-machine-learning)

### Code example: Gaussian Mixture Models

Gaussian mixture models (GMM) hain a generative model that assumes the data has been generated by a mix ka various multivariate normal distributions. The algorithm aims to estimate the mean & covariance matrices ka these distributions.

It generalizes the k-Means algorithm: it adds covariance among features so that clusters can be ellipsoids rather than spheres, while the centroids hain represented by the means ka each distribution. The GMM algorithm performs soft assignments because each point has a probability to be a member ka any cluster. 

Notebook [gaussian_mixture_models](03_clustering_algorithms/06_gaussian_mixture_models.ipynb) demonstrate karta hai the application ka a GMM clustering model.

### Code example: Hierarchical Risk Parity

The key idea ka hierarchical risk parity (HRP) hai to use hierarchical clustering on the covariance matrix to be able to group assets ke saath similar correlations together aur reduce the number ka degrees ka freedom by only considering 'similar' assets as substitutes when constructing the portfolio. 

#### The algorithm

Notebook [hierarchical_risk_parity](04_hierarchical_risk_parity/01_hierarchical_risk_parity.ipynb) mein the subfolder [hierarchical_risk_parity](04_hierarchical_risk_parity) illustrate its application. 

#### Backtest comparison ke saath alternatives

Notebook [pf_optimization_with_hrp_zipline_benchmark](04_hierarchical_risk_parity/02_pf_optimization_with_hrp_zipline_benchmark.ipynb) mein the subfolder [hierarchical_risk_parity](04_hierarchical_risk_parity) compares HRP ke saath other portfolio optimization methods discussed mein [Chapter 5](../05_strategy_evaluation). 

### References

- [Building Diversified Portfolios that Outperform Out-ka-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678), Lopez de Prado, Journal ka Portfolio Management, 2015
- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Raffinot 2016
- [Visualizing the Stock Market Structure](https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html)



