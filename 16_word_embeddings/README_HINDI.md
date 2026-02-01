# Word Embeddings ke liye Earnings Calls aur SEC Filings 

Yeh chapter introduces use karta hai neural networks to learn a vector representation ka individual semantic units like a word or a paragraph. These vectors hain dense rather than sparse as mein the bag-ka-words model aur have a few hundred real-valued rather than tens ka thousand binary or discrete entries. They hain called embeddings because they assign each semantic unit a location mein a continuous vector space.
 
Embeddings result from training a model to relate tokens to their context ke saath the benefit that similar usage implies a similar vector. As a result, the embeddings encode semantic aspects like relationships among words by means ka their relative location. They hain powerful features ke liye use mein the deep learning models that hum will introduce mein the following chapters.


## Vishay-suchi (Content)

1. [How Word Embeddings encode Semantics](#how-word-embeddings-encode-semantics)
    * [How neural language models learn usage in context](#how-neural-language-models-learn-usage-in-context)
    * [The word2vec Model: scalable word and phrase embeddings](#the-word2vec-model-scalable-word-and-phrase-embeddings)
    * [Evaluating embeddings: vector arithmetic and analogies](#evaluating-embeddings-vector-arithmetic-and-analogies)
2. [Code example: Working ke saath embedding models](#code-example-working-ke saath-embedding-models)
    * [Working with Global Vectors for Word Representation (GloVe)](#working-with-global-vectors-for-word-representation-glove)
    * [Evaluating embeddings using analogies](#evaluating-embeddings-using-analogies)
3. [Code example: training domain-specific embeddings use karke financial news](#code-example-training-domain-specific-embeddings-use karke-financial-news)
    * [Preprocessing financial news: sentence detection and n-grams](#preprocessing-financial-news-sentence-detection-and-n-grams)
    * [Skip-gram architecture in TensorFlow 2 and visualization with TensorBoard](#skip-gram-architecture-in-tensorflow-2-and-visualization-with-tensorboard)
    * [How to train embeddings faster with Gensim](#how-to-train-embeddings-faster-with-gensim)
4. [Code Example: word Vectors from SEC Filings use karke gensim](#code-example-word-vectors-from-sec-filings-use karke-gensim)
    * [Preprocessing: content selection, sentence detection, and n-grams](#preprocessing-content-selection-sentence-detection-and-n-grams)
    * [Model training and evaluation](#model-training-and-evaluation)
5. [Code example: sentiment Analysis ke saath Doc2Vec](#code-example-sentiment-analysis-ke saath-doc2vec)
6. [New Frontiers: Attention, Transformers, aur Pretraining](#new-frontiers-attention-transformers-aur-pretraining)
    * [Attention is all you need: transforming natural language generation](#attention-is-all-you-need-transforming-natural-language-generation)
    * [BERT: Towards a more universal, pretrained language model](#bert-towards-a-more-universal-pretrained-language-model)
    * [Using pretrained state-of-the-art models](#using-pretrained-state-of-the-art-models)
7. [Additional Resources](#additional-resources)

## How Word Embeddings encode Semantics

Word embeddings represent tokens as lower-dimensional vectors so that their relative location reflects their relationship mein terms ka how they hain used mein context. They embody the distributional hypothesis from linguistics that claims words hain best defined by the company they keep.

Word vectors hain capable ka capturing numerous semantic aspects; not only hain synonyms close to each other, but words can have multiple degrees ka similarity, e.g. the word ‘driver’ could be similar to ‘motorist’ or to ‘factor’. Furthermore, embeddings reflect relationships among pairs ka words like analogies (Tokyo hai to Japan what Paris hai to France, or went hai to go what saw hai to see).  

### How neural language models learn usage mein context

Word embeddings result from a training a shallow neural network to predict a word given its context. Whereas traditional language models define context as the words preceding the target, word embedding models use the words contained mein a symmetric window surrounding the target. 

mein contrast, the bag-ka-words model use karta hai the entire documents as context aur use karta hai (weighted) counts to capture the co-occurrence ka words rather than predictive vectors.

### The word2vec Model: scalable word aur phrase embeddings

A word2vec model hai a two-layer neural net that takes a text corpus as input aur outputs a set ka embedding vectors ke liye words mein that corpus. There hain two different architectures to efficiently learn word vectors use karke shallow neural networks.
- The **continuous-bag-ka-words** (CBOW) model predicts the target word use karke the average ka the context word vectors as input so that their order does not matter. CBOW trains faster aur tends to be slightly more accurate ke liye frequent terms, but pays less attention to infrequent words.
- The **skip-gram** (SG) model, mein contrast, use karta hai the target word to predict words sampled from the context. It works well ke saath small datasets aur finds good representations even ke liye rare words or phrases.

### Evaluating embeddings: vector arithmetic aur analogies

The dimensions ka the word aur phrase vectors do not have an explicit meaning. However, the embeddings encode similar usage as proximity mein the latent space mein a way that carries over to semantic relationships. Yeh results mein the interesting properties that analogies can be expressed by adding aur subtracting word vectors.

Just as words can be used mein different contexts, they can be related to other words mein different ways, aur these relationships correspond to different directions mein the latent space. Accordingly, there hain several types ka analogies that the embeddings should reflect if the training data permits.

The word2vec authors provide a list ka several thousand relationships spanning aspects ka geography, grammar aur syntax, aur family relationships to evaluate the quality ka embedding vectors (see directory [analogies](data/analogies)).

## Code example: Working ke saath embedding models

Similar to other unsupervised learning techniques, the goal ka learning embedding vectors hai to generate features ke liye other tasks like text classification or sentiment analysis.
There hain several options to obtain embedding vectors ke liye a given corpus ka documents:
- Use embeddings learned from a generic large corpus like Wikipedia or Google News
- Train your own model use karke documents that reflect a domain ka interest

### Working ke saath Global Vectors ke liye Word Representation (GloVe)

GloVe hai an unsupervised algorithm developed at the Stanford NLP lab that learns vector representations ke liye words from aggregated global word-word co-occurrence statistics (see references). Vectors pre-trained on the following web-scale sources hain available:
- Common Crawl ke saath 42B or 840B tokens aur a vocabulary ka 1.9M or 2.2M tokens
- Wikipedia 2014 + Gigaword 5 ke saath 6B tokens aur a vocabulary ka 400K tokens
- Twitter use karke 2B tweets, 27B tokens aur a vocabulary ka 1.2M tokens

The following table shows the accuracy on the word2vec semantics test achieved by the GloVE vectors trained on Wikipedia:

| Category                 | Samples | Accuracy | Category              | Samples | Accuracy |
|--------------------------|---------|----------|-----------------------|---------|----------|
| capital-common-countries | 506     | 94.86%   | comparative           | 1332    | 88.21%   |
| capital-world            | 8372    | 96.46%   | superlative           | 1056    | 74.62%   |
| city-mein-state            | 4242    | 60.00%   | present-participle    | 1056    | 69.98%   |
| currency                 | 752     | 17.42%   | nationality-adjective | 1640    | 92.50%   |
| family                   | 506     | 88.14%   | past-tense            | 1560    | 61.15%   |
| adjective-to-adverb      | 992     | 22.58%   | plural                | 1332    | 78.08%   |
| opposite                 | 756     | 28.57%   | plural-verbs          | 870     | 58.51%   |

There hain several sources ke liye pre-trained word embeddings. Popular options include Stanford’s GloVE aur spaCy’s built-mein vectors.
- Notebook [using_trained_vectors ](01_using_trained_vectors.ipynb) illustrates how to work ke saath pretrained vectors.

### Evaluating embeddings use karke analogies

Notebook [evaluating_embeddings](02_evaluating_embeddings.ipynb) demonstrate karta hai how to test the quality ka word vectors use karke analogies aur other semantic relationships among words.

## Code example: training domain-specific embeddings use karke financial news

Many tasks require embeddings ka domain-specific vocabulary that models pre-trained on a generic corpus may not be able to capture. Standard word2vec models hain not able to assign vectors to out-ka-vocabulary words aur instead use a default vector that reduces their predictive value. 

Udaharan ke liye, when working ke saath industry-specific documents, the vocabulary or its usage may change over time as new technologies or products emerge. As a result, the embeddings need to evolve as well. mein addition, documents like corporate earnings releases use nuanced language that GloVe vectors pre-trained on Wikipedia articles hain unlikely to properly reflect.

See the [data](../data) directory ke liye instructions on sourcing the financial news dataset.

### Preprocessing financial news: sentence detection aur n-grams

Notebook [financial_news_preprocessing](03_financial_news_preprocessing.ipynb) demonstrate karta hai how to prepare the source data ke liye our model

### Skip-gram architecture mein TensorFlow 2 aur visualization ke saath TensorBoard

Notebook [financial_news_word2vec_tensorflow](04_financial_news_word2vec_tensorflow.ipynb) illustrates how to build a word2vec model use karke the Keras interface ka TensorFlow 2 that hum will introduce mein much more detail mein the next chapter. 

### How to train embeddings faster ke saath Gensim

The TensorFlow implementation hai very transparent mein terms ka its architecture, but it hai not particularly fast. The natural language processing (NLP) library [gensim](https://radimrehurek.com/gensim/) that hum also used ke liye topic modeling mein the last chapter, offers better performance aur more closely resembles the C-based word2vec implementation provided by the original authors.

Notebook [financial_news_word2vec_gensim](05_financial_news_word2vec_gensim.ipynb) shows how to learn word vectors more efficiently.

## Code Example: word Vectors from SEC Filings use karke gensim

mein this section, hum will learn word aur phrase vectors from annual SEC filings use karke gensim to illustrate the potential value ka word embeddings ke liye algorithmic trading. mein the following sections, hum will combine these vectors as features ke saath price returns to train neural networks to predict equity prices from the content ka security filings.

mein particular, hum use a dataset containing over 22,000 10-K annual reports from the period 2013-2016 that hain filed by listed companies aur contain both financial information aur management commentary (see Chapter 3 on [Alternative Data](../03_alternative_data)). ke liye about half ka 11K filings ke liye companies that hum have stock prices to label the data ke liye predictive modeling (see references about data source aur the notebooks mein the folder [sec-filings](sec-filings) ke liye details). 

- [2013-2016 Cleaned/Parsed 10-K Filings ke saath the SEC](https://data.world/jumpyaf/2013-2016-cleaned-parsed-10-k-filings-ke saath-the-sec)
- [Stock Market Predictions ke saath Natural Language Deep Learning](https://www.microsoft.com/developerblog/2017/12/04/predicting-stock-performance-deep-learning/)

### Preprocessing: content selection, sentence detection, aur n-grams

Notebook [sec_preprocessing](06_sec_preprocessing.ipynb) shows how to parse aur tokenize the text use karke spaCy, similar to the approach mein Chapter 14, [Text Data ke liye Trading: Sentiment Analysis](../14_working_with_text_data). 

### Model training aur evaluation

Notebook [sec_word2vec](07_sec_word2vec.ipynb) use karta hai gensim's [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) implementation ka the skip-gram architecture to learn word vectors ke liye the SEC filings dataset.

## Code example: sentiment Analysis ke saath Doc2Vec

Text classification requires combining multiple word embeddings. A common approach hai to average the embedding vectors ke liye each word mein the document. Yeh use karta hai information from all embeddings aur effectively use karta hai vector addition to arrive at a different location point mein the embedding space. However, relevant information about the order ka words hai lost. 

mein contrast, the state-ka-the-art generation ka embeddings ke liye pieces ka text like a paragraph or a product review hai to use the document embedding model doc2vec. Yeh model was developed by the word2vec authors shortly after publishing their original contribution. Similar to word2vec, there hain also two flavors ka doc2vec:
- The distributed bag ka words (DBOW) model corresponds to the Word2Vec CBOW model. The document vectors result from training a network on the synthetic task ka predicting a target word based on both the context word vectors aur the document's doc vector.
- The distributed memory (DM) model corresponds to the word2wec skipgram architecture. The doc vectors result from training a neural net to predict a target word use karke the full document’s doc vector.

Notebook [doc2vec_yelp_sentiment](08_doc2vec_yelp_sentiment.ipynb) applies doc2vec to a random sample ka 1mn Yelp reviews ke saath their associated star ratings.

## New Frontiers: Attention, Transformers, aur Pretraining

Word2vec aur GloVe embeddings capture more semantic information than the bag-ka-words approach, but only allow ke liye a single fixed-length representation ka each token that does not differentiate between context-specific usages. To address unsolved problems like multiple meanings ke liye the same word, called polysemy, several new models have emerged that build on the attention mechanism designed to learn more contextualized word embeddings ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)). Key characteristics ka these models hain 
- the use ka bidirectional language models that process text both left-to-right aur right-to-left ke liye a richer context representation, aur
- the use ka semi-supervised pretraining on a large generic corpus to learn universal language aspects mein the form ka embeddings aur network weights that can be used end fine-tuned ke liye specific tasks

### Attention hai all you need: transforming natural language generation

mein 2018, Google released the BERT model, which stands ke liye Bidirectional Encoder Representations from Transformers ([Devlin et al. 2019](https://arxiv.org/abs/1810.04805)). mein a major breakthrough ke liye NLP research, it achieved groundbreaking results on eleven natural language understanding tasks ranging from question answering aur named entity recognition to paraphrasing aur sentiment analysis as measured by the General Language Understanding Evaluation (GLUE) [benchmark](https://gluebenchmark.com/).

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Visualizing A Neural Machine Translation Model (Mechanics ka Seq2seq Models ke saath Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-ka-seq2seq-models-ke saath-attention/)

### BERT: Towards a more universal, pretrained language model

The BERT model builds on two key ideas, namely the transformer architecture described mein the previous section aur unsupervised pre-training so that it doesn’t need to be trained from scratch ke liye each new task; rather, its weights hain fine-tuned.
- BERT takes the attention mechanism to a new (deeper) level by use karke 12 or 24 layers depending on the architecture, each ke saath 12 or 16 attention heads, resulting mein up to 24 x 16 = 384 attention mechanisms to learn context-specific embeddings.  
- BERT use karta hai unsupervised, bidirectional pre-training to learn its weights mein advance on two tasks: masked language modeling (predicting a missing word given the left aur right context) aur next sentence prediction (predicting whether one sentence follows another).

- [The Illustrated BERT, ELMo, aur co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)
- [The General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/leaderboard)
- [Financial NLP at S&P Global ](https://www.youtube.com/watch?v=rdmaR4WRYEM&list=PLBmcuObd5An4UC6jvK_-eSl6jCvP1gwXc&index=9)

### use karke pretrained state-ka-the-art models

- [Huggingface Transformers](https://github.com/huggingface/transformers)
    - Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, T5, CTRL...) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over thousands of pretrained models in 100+ languages and deep interoperability between PyTorch & TensorFlow 2.0.
- [spacy-transformers](https://github.com/explosion/spacy-transformers)
    - This package (previously spacy-pytorch-transformers) provides spaCy model pipelines that wrap Hugging Face's transformers package, so you can use them in spaCy. The result is convenient access to state-of-the-art transformer architectures, such as BERT, GPT-2, XLNet, etc. For more details and background.
- [Allen NLP](https://allennlp.org/)
    - Deep learning for NLP: AllenNLP makes it easy to design and evaluate new deep learning models for nearly any NLP problem, along with the infrastructure to easily run them in the cloud or on your laptop.
    - State of the art models: AllenNLP includes reference implementations of high quality models for both core NLP problems (e.g. semantic role labeling) and NLP applications (e.g. textual entailment).
- [Sentence Transformers: Multilingual Sentence Embeddings use karke BERT / RoBERTa / XLM-RoBERTa & Co. ke saath PyTorch]
     -BERT / RoBERTa / XLM-RoBERTa produces out-of-the-box rather bad sentence embeddings. This repository fine-tunes BERT / RoBERTa / DistilBERT / ALBERT / XLNet with a siamese or triplet network structure to produce semantically meaningful sentence embeddings that can be used in unsupervised scenarios: Semantic textual similarity via cosine-similarity, clustering, semantic search.

## Additional Resources

- [GloVe: Global Vectors ke liye Word Representation](https://github.com/stanfordnlp/GloVe)
- [Common Crawl Data](http://commoncrawl.org/the-data/)
- [word2vec analogy samples](https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt)
- [spaCy word vectors aur semantic similarity](https://spacy.io/usage/vectors-similarity)
- [2013-2016 Cleaned/Parsed 10-K Filings ke saath the SEC](https://data.world/jumpyaf/2013-2016-cleaned-parsed-10-k-filings-ke saath-the-sec)
- [Stanford Sentiment Tree Bank](https://nlp.stanford.edu/sentiment/treebank.html)
- [Word embeddings | TensorFlow Core](https://www.tensorflow.org/tutorials/text/word_embeddings)
- [Visualizing Data use karke the Embedding Projector mein TensorBoard](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin)
