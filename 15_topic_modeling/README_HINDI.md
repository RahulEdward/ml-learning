# Topic Modeling ke liye Earnings Calls aur Financial News

Yeh chapter use karta hai unsupervised machine learning to extract latent topics from documents. These themes can produce detailed insights into a large body ka documents mein an automated way. They hain very useful to understand the haystack itself aur permit the concise tagging ka documents because use karke the degree ka association ka topics aur documents. 

Topic models permit the extraction ka sophisticated, interpretable text features that can be used mein various ways to extract trading signals from large collections ka documents. They speed up the review ka documents, help identify aur cluster similar documents, aur can be annotated as a basis ke liye predictive modeling. Applications include the identification ka key themes mein company disclosures or earnings call transcripts, customer reviews or contracts, annotated use karke, e.g., sentiment analysis or direct labeling ke saath subsequent asset returns. More specifically, hai chapter mein, hum cover karenge:

## Vishay-suchi (Content)

1. [Learning latent topics: goals aur approaches](#learning-latent-topics-goals-aur-approaches)
2. [Latent semantic indexing (LSI)](#latent-semantic-indexing-lsi)
    * [Code example: how to implement LSI using scikit-learn](#code-example-how-to-implement-lsi-using-scikit-learn)
3. [Probabilistic Latent Semantic Analysis (pLSA)](#probabilistic-latent-semantic-analysis-plsa)
    * [Code example: how to implement pLSA using scikit-learn](#code-example-how-to-implement-plsa-using-scikit-learn)
4. [Latent Dirichlet Allocation (LDA)](#latent-dirichlet-allocation-lda)
    * [Code example: the Dirichlet distribution](#code-example-the-dirichlet-distribution)
    * [How to evaluate LDA topics](#how-to-evaluate-lda-topics)
    * [Code example: how to implement LDA using scikit-learn](#code-example-how-to-implement-lda-using-scikit-learn)
    * [How to visualize LDA results using pyLDAvis](#how-to-visualize-lda-results-using-pyldavis)
    * [Code example: how to implement LDA using gensim](#code-example-how-to-implement-lda-using-gensim)
    * [References](#references)
5. [Code example: Modeling topics discussed during earnings calls](#code-example-modeling-topics-discussed-during-earnings-calls)
6. [Code example: topic modeling ke saath financial news articles](#code-example-topic-modeling-ke saath-financial-news-articles)
7. [Resources](#resources)
    * [Applications](#applications)
    * [Topic Modeling libraries](#topic-modeling-libraries)

## Learning latent topics: goals aur approaches

Initial attempts by topic models to improve on the vector space model (developed mein the mid-1970s) applied linear algebra to reduce the dimensionality ka the document-term matrix. Yeh approach hai similar to the algorithm hum discussed as principal component analysis mein chapter 12 on unsupervised learning. While effective, it hai difficult to evaluate the results ka these models absent a benchmark model.

mein response, probabilistic models emerged that assume an explicit document generation process aur provide algorithms to reverse engineer this process aur recover the underlying topics.

The below table highlights key milestones mein the model evolution that hum will address mein more detail mein the following sections.

| Model                                         | Year | Description                                                                                                   |
|-----------------------------------------------|------|---------------------------------------------------------------------------------------------------------------|
| Latent Semantic Indexing (LSI)                | 1988 | Capture semantic document-term relationship by reducing the dimensionality ka the word space                  |
| Probabilistic Latent Semantic Analysis (pLSA) | 1999 | Reverse-engineers a generative a process that assumes words generate a topic aur documents as a mix ka topics |
| Latent Dirichlet Allocation (LDA)             | 2003 | Adds a generative process ke liye documents: three-level hierarchical, Bayesian model                             |

## Latent semantic indexing (LSI)

Latent Semantic Analysis set out to improve the results ka queries that omitted relevant documents containing synonyms ka query terms. Its aimed to model the relationships between documents aur terms to be able to predict that a term should be associated ke saath a document, even though, because ka variability mein word use, no such association was observed.

LSI use karta hai linear algebra to find a given number k ka latent topics by decomposing the DTM. More specifically, it use karta hai the Singular Value Decomposition (SVD) to find the best lower-rank DTM approximation use karke k singular values & vectors. mein other words, LSI hai an application ka the unsupervised learning techniques ka dimensionality reduction hum encountered mein chapter 12 (ke saath some additional detail). The authors experimented ke saath hierarchical clustering but found it too restrictive to explicitly model the document-topic aur topic-term relationships or capture associations ka documents or terms ke saath several topics.

### Code example: how to implement LSI use karke scikit-learn

Notebook [latent_semantic_indexing](01_latent_semantic_indexing.ipynb) demonstrate karta hai how to apply LSI to the BBC new articles hum used mein the last chapter.

## Probabilistic Latent Semantic Analysis (pLSA)

Probabilistic Latent Semantic Analysis (pLSA) takes a statistical perspective on LSA aur create karta hai a generative model to address the lack ka theoretical underpinnings ka LSA. 

pLSA explicitly models the probability each co-occurrence ka documents d aur words w described by the DTM as a mixture ka conditionally independent multinomial distributions that involve topics t. The number ka topics hai a hyperparameter chosen prior to training aur hai not learned from the data.

### Code example: how to implement pLSA use karke scikit-learn
 
Notebook [probabilistic_latent_analysis](02_probabilistic_latent_analysis.ipynb) demonstrate karta hai how to apply LSI to the BBC new articles hum used mein the last chapter.

- [Relation between PLSA aur NMF aur Implications](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.8839&rep=rep1&type=pdf), Gaussier, Goutte, 2005

## Latent Dirichlet Allocation (LDA)

Latent Dirichlet Allocation extends pLSA by adding a generative process ke liye topics.
It hai the most popular topic model because it tends to produce meaningful topics that humans, can relate to, can assign topics to new documents, aur hai extensible. Variants ka LDA models can include metadata like authors, or include image data, or learn hierarchical topics.

LDA hai a hierarchical Bayesian model that assumes topics hain probability distributions over words, aur documents hain distributions over topics. More specifically, the model assumes that topics follow a sparse Dirichlet distribution, which implies that documents cover only a small set ka topics, aur topics use only a small set ka words frequently. 

### Code example: the Dirichlet distribution

The Dirichlet distribution produces probability vectors that can be used ke saath discrete distributions. That hai, it randomly generates a given number ka values that hain positive aur sum to one. It has a parameter 𝜶 ka positive real value that controls the concentration ka the probabilities.

Notebook [dirichlet_distribution](03_dirichlet_distribution.ipynb) contain karta hai a simulation so you can experiment ke saath different parameter values.

### How to evaluate LDA topics

Unsupervised topic models do not provide a guarantee that the result will be meaningful or interpretable, aur there hai no objective metric to assess the result as mein supervised learning. Human topic evaluation hai considered the ‘gold standard’ but hai potentially expensive aur not readily available at scale.

Two options to evaluate results more objectively include perplexity that evaluates the model on unseen documents aur topic coherence metrics that aim to evaluate the semantic quality ka the uncovered patterns.

### Code example: how to implement LDA use karke scikit-learn

Notebook [lda_with_sklearn](04_lda_with_sklearn.ipynb) shows how to apply LDA to the BBC news articles. hum use [sklearn.decomposition.LatentDirichletAllocation](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) to train an LDA model ke saath five topics.

### How to visualize LDA results use karke pyLDAvis

Topic visualization facilitates the evaluation ka topic quality use karke human judgment. pyLDAvis hai a python port ka LDAvis, developed mein R aur D3.js. hum will introduce the key concepts; each LDA implementation notebook contain karta hai examples.

pyLDAvis displays the global relationships among topics while also facilitating their semantic evaluation by inspecting the terms most closely associated ke saath each individual topic aur, inversely, the topics associated ke saath each term. It also addresses the challenge that terms that hain frequent mein a corpus tend to dominate the multinomial distribution over words that define a topic. LDAVis introduces the relevance r ka term w to topic t to produce a flexible ranking ka key terms use karke a weight parameter 0<=ƛ<=1. 

- [Talk by the Author](https://speakerdeck.com/bmabey/visualizing-topic-models) aur [Paper by (original) Author](http://www.aclweb.org/anthology/W14-3110)
- [Documentation](http://pyldavis.readthedocs.io/en/latest/index.html)

### Code example: how to implement LDA use karke gensim

Gensim hai a specialized NLP library ke saath a fast LDA implementation aur many additional features. hum will also use it mein the next chapter to learn word vectors (see the notebook [lda_with_gensim](05_lda_with_gensim.ipynb) ke liye details.

### References

- [David Blei Homepage @ Columbia](http://www.cs.columbia.edu/~blei/)
- [Introductory Paper](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf) aur [more technical review paper](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2009.pdf)
- [Blei Lab @ GitHub](https://github.com/Blei-Lab)
- [Exploring Topic Coherence over many models aur many topics](https://www.aclweb.org/anthology/D/D12/D12-1087.pdf)
- [Paper on various Methods](http://www.aclweb.org/anthology/N10-1012)
- [Blog Post - Overview](http://qpleple.com/topic-coherence-to-evaluate-topic-models/)

## Code example: Modeling topics discussed during earnings calls

mein Chapter 3 on [Alternative Data](../03_alternative_data/02_earnings_calls), hum learned how to scrape earnings call data from the SeekingAlpha site. 

mein this section, hum will illustrate topic modeling use karke this source. I’m use karke a sample ka some 700 earnings call transcripts from 2018 aur 2019 (see [data](../data) directory). Yeh hai a fairly small sample; ke liye a practical application, hum would need a larger dataset.  
 
Notebook [lda_earnings_calls](06_lda_earnings_calls.ipynb) contain karta hai details on loading, exploring, aur preprocessing the data, as well as training aur evaluating different models.

## Code example: topic modeling ke saath financial news articles

Notebook [lda_financial_news](07_lda_financial_news.ipynb) shows how to summarize a large corpus ka financial news articles sourced from Reuters aur others (see [data](../data) ke liye sources) use karke LDA.

## Sansadhan (Resources)

### Applications

- [Applications ka Topic Models](https://mimno.infosci.cornell.edu/papers/2017_fntir_tm_applications.pdf), Jordan Boyd-Graber, Yuening Hu, David Mimno, 2017
- [High Quality Topic Extraction from Business News Explains Abnormal Financial Market Volatility](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3675119/pdf/pone.0064846.pdf)
- [What hain You Saying? use karke Topic to Detect Financial Misreporting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2803733)
- [LDA mein the browser - javascript implementation](https://github.com/mimno/jsLDA)
- [David Mimno @ Cornell University](https://mimno.infosci.cornell.edu/)

### Topic Modeling libraries

- [David Blei's List ka Open Source Topic Modeling Software](http://www.cs.columbia.edu/~blei/topicmodeling_software.html)
- [Mallet (MAchine Learning ke liye LanguagE Toolkit (mein Java)](http://mallet.cs.umass.edu/)
