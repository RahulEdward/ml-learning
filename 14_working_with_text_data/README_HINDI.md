# Text Data ke liye Trading: Sentiment Analysis

Yeh hai the first ka three chapters dedicated to extracting signals ke liye algorithmic trading strategies from text data use karke natural language processing (NLP) aur machine learning.

Text data hai very rich mein content but highly unstructured so that it requires more preprocessing to enable an ML algorithm to extract relevant information. A key challenge consists ka converting text into a numerical format without losing its meaning. hum will cover several techniques capable ka capturing nuances ka language so that they can be used as input ke liye ML algorithms.

mein this chapter, hum will introduce fundamental feature extraction techniques that focus on individual semantic units, i.e. words or short groups ka words called tokens. hum will show how to represent documents as vectors ka token counts by creating a document-term matrix aur then proceed to use it as input ke liye news classification aur sentiment analysis. hum will also introduce the Naive Bayes algorithm that hai popular ke liye this purpose.

mein the following two chapters, hum build on these techniques aur use ML algorithms like topic modeling aur word-vector embeddings to capture the information contained mein a broader context. 

## Vishay-suchi (Content)

1. [ML ke saath text data - from language to features](#ml-ke saath-text-data---from-language-to-features)
    * [Challenges of Natural Language Processing](#challenges-of-natural-language-processing)
    * [Use cases](#use-cases)
    * [The NLP workflow](#the-nlp-workflow)
2. [From text to tokens – the NLP pipeline](#from-text-to-tokens--the-nlp-pipeline)
    * [Code example: NLP pipeline with spaCy and textacy](#code-example-nlp-pipeline-with-spacy-and-textacy)
        - [Data](#data)
    * [Code example: NLP with TextBlob](#code-example-nlp-with-textblob)
3. [Counting tokens – the document-term matrix](#counting-tokens--the-document-term-matrix)
    * [Code example: document-term matrix with scikit-learn](#code-example-document-term-matrix-with-scikit-learn)
4. [NLP ke liye trading: text classification aur sentiment analysis](#nlp-ke liye-trading-text-classification-aur-sentiment-analysis)
    * [The Naive Bayes classifier](#the-naive-bayes-classifier)
    * [Code example: news article classification](#code-example-news-article-classification)
    * [Code examples: sentiment analysis](#code-examples-sentiment-analysis)
        - [Binary classification: twitter data](#binary-classification-twitter-data)
        - [Comparing different ML algorithms on large, multiclass Yelp data](#comparing-different-ml-algorithms-on-large-multiclass-yelp-data)

## ML ke saath text data - from language to features

Text data can be extremely valuable given how much information humans communicate aur store use karke natural language. The diverse set ka data sources relevant to investment range from formal documents like company statements, contracts, or patents to news, opinion, aur analyst research or commentary to various types ka social media postings or messages. 

Useful resources include:

- [Speech aur Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf), Daniel Jurafsky & James H. Martin, 3rd edition, draft, 2018
- [Statistical natural language processing aur corpus-based computational linguistics](https://nlp.stanford.edu/links/statnlp.html), Annotated list ka resources, Stanford University
- [NLP Data Sources](https://github.com/niderhoff/nlp-datasets)

### Challenges ka Natural Language Processing

The conversion ka unstructured text into a machine-readable format requires careful preprocessing to preserve the valuable semantic aspects ka the data. How humans derive meaning from aur comprehend the content ka language hai not fully understood aur improving language understanding by machines remains an area ka very active research. 

NLP hai challenging because the effective use ka text data ke liye machine learning requires an understanding ka the inner workings ka language as well as knowledge about the world to which it refers. Key challenges include:
- ambiguity due to polysemy, i.e. a word or phrase can have different meanings that depend on context (‘Local High School Dropouts Cut mein Half’)
- non-standard aur evolving use ka language, especially mein social media
- idioms: ‘throw mein the towel’
- entity names can be tricky : ‘Where hai A Bug's Life playing?’
- the need ke liye knowledge about the world: ‘Mary aur Sue hain sisters’ vs ‘Mary aur Sue hain mothers’

### Use cases

| Use Case  | Description  | Examples  |
|---|---|---|
| Chatbots | Understand natural language from the user aur return intelligent responses | [Api.ai](https://api.ai/) |
| Information retrieval | Find relevant results aur similar results | [Google](https://www.google.com/) |
| Information extraction | Structured information from unstructured documents | [Events from Gmail](https://support.google.com/calendar/answer/6084018?hl=en) |
| Machine translation | One language to another | [Google Translate](https://translate.google.com/) |
| Text simplification | Preserve the meaning ka text, but simplify the grammar aur vocabulary | [Rewordify](https://rewordify.com/), [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) |
| Predictive text input | Faster or easier typing  | [Phrase completion](https://justmarkham.shinyapps.io/textprediction/), [A much better application](https://farsite.shinyapps.io/swiftkey-cap/) |
| Sentiment analysis | Attitude ka speaker | [Hater News](https://medium.com/@KevinMcAlear/building-hater-news-62062c58325c) |
| Automatic summarization | Extractive or abstractive summarization | [reddit's autotldr algo](https://smmry.com/about), [autotldr example](https://www.reddit.com/r/technology/comments/35brc8/21_million_people_still_use_aol_dialup/cr2zzj0)  |
| Natural language generation | Generate text from data | [How a computer describes a sports match](http://www.bbc.com/news/technology-34204052), [Publishers withdraw more than 120 gibberish papers](http://www.nature.com/news/publishers-withdraw-more-than-120-gibberish-papers-1.14763) |
| Speech recognition aur generation | Speech-to-text, text-to-speech | [Google's Web Speech API demo](https://www.google.com/intl/en/chrome/demos/speech.html), [Vocalware Text-to-Speech demo](https://www.vocalware.com/index/demo) |
| Question answering | Determine the intent ka the question, match query ke saath knowledge base, evaluate hypotheses | [How did Watson beat Jeopardy champion Ken Jennings?](http://blog.ted.com/how-did-supercomputer-watson-beat-jeopardy-champion-ken-jennings-experts-discuss/), [Watson Trivia Challenge](http://www.nytimes.com/interactive/2010/06/16/magazine/watson-trivia-game.html), [The AI Behind Watson](http://www.aaai.org/Magazine/Watson/watson.php)

### The NLP workflow

A key goal ke liye use karke machine learning from text data ke liye algorithmic trading hai to extract signals from documents. A document hai an individual sample from a relevant text data source, e.g. a company report, a headline or news article, or a tweet. A corpus, mein turn, hai a collection ka documents.
The following figure lays out key steps to convert documents into a dataset that can be used to train a supervised machine learning algorithm capable ka making actionable predictions.

<p align="center">
<img src="https://i.imgur.com/LPxpc8D.png" width="90%">
</p>

## From text to tokens – the NLP pipeline

The following table summarizes the key tasks ka an NLP pipeline:

| Feature                     | Description                                                       |
|-----------------------------|-------------------------------------------------------------------|
| Tokenization                | Segment text into words, punctuations marks etc.                  |
| Part-ka-speech tagging      | Assign word types to tokens, like a verb or noun.                 |
| Dependency parsing          | Label syntactic token dependencies, like subject <=> object.      |
| Stemming & Lemmatization    | Assign the base forms ka words: "was" => "be", "rats" => "rat".   |
| Sentence boundary detection | Find aur segment individual sentences.                            |
| Named Entity Recognition    | Label "real-world" objects, like persons, companies or locations. |
| Similarity                  | Evaluate similarity ka words, text spans, aur documents.          |

### Code example: NLP pipeline ke saath spaCy aur textacy

Notebook [nlp_pipeline_with_spaCy](01_nlp_pipeline_with_spaCy.ipynb) demonstrate karta hai how to construct an NLP pipeline use karke the open-source python library [spaCy]((https://spacy.io/)). The [textacy](https://chartbeat-labs.github.io/textacy/index.html) library builds on spaCy aur provide karta hai easy access to spaCy attributes aur additional functionality.

- spaCy [docs](https://spacy.io/) aur installation [instructions](https://spacy.io/usage/#installation)
- textacy relies on `spaCy` to solve additional NLP tasks - see [documentation](https://chartbeat-labs.github.io/textacy/index.html)

#### Data
- [BBC Articles](http://mlg.ucd.ie/datasets/bbc.html), use raw text files
- [TED2013](http://opus.nlpl.eu/TED2013.php), a parallel corpus ka TED talk subtitles mein 15 languages

### Code example: NLP ke saath TextBlob

The `TextBlob` library provide karta hai a simplified interface ke liye common NLP tasks including part-ka-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, aur others.

Notebook [nlp_with_textblob](02_nlp_with_textblob.ipynb) illustrates its functionality.

- [Documentation](https://textblob.readthedocs.io/en/dev/)
- [Sentiment Analysis](https://github.com/sloria/TextBlob/blob/dev/textblob/en/en-sentiment.xml)

A good alternative hai NLTK, a leading platform ke liye building Python programs to work ke saath human language data. It provide karta hai easy-to-use interfaces to over 50 corpora aur lexical resources such as WordNet, along ke saath a suite ka text processing libraries ke liye classification, tokenization, stemming, tagging, parsing, aur semantic reasoning, wrappers ke liye industrial-strength NLP libraries, aur an active discussion forum.

- Natural Language ToolKit (NLTK) [Documentation](http://www.nltk.org/)

## Counting tokens – the document-term matrix

Yeh section introduces the bag-ka-words model that converts text data into a numeric vector space representation that permits the comparison ka documents use karke their distance. hum demonstrate how to create a document-term matrix use karke the sklearn library.

- [TF-IDF hai about what matters](https://planspace.org/20150524-tfidf_is_about_what_matters/)

### Code example: document-term matrix ke saath scikit-learn

The scikit-learn preprocessing module offers two tools to create a document-term matrix. 
1. The [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) use karta hai binary or absolute counts to measure the term frequency tf(d, t) ke liye each document d aur token t.
2. The [TfIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), mein contrast, weighs the (absolute) term frequency by the inverse document frequency (idf). As a result, a term that appears mein more documents will receive a lower weight than a token ke saath the same frequency ke liye a given document but lower frequency across all documents

Notebook [document_term_matrix](03_document_term_matrix.ipynb) demonstrate usage aur configuration.

## NLP ke liye trading: text classification aur sentiment analysis

Once text data has been converted into numerical features use karke the natural language processing techniques discussed mein the previous sections, text classification works just like any other classification task.

mein this section, hum will apply these preprocessing technique to news articles, product reviews, aur Twitter data aur teach various classifiers to predict discrete news categories, review scores, aur sentiment polarity.

First, hum will introduce the Naive Bayes model, a probabilistic classification algorithm that works well ke saath the text features produced by a bag-ka-words model.

- [Daily Market News Sentiment aur Stock Prices](https://www.econstor.eu/handle/10419/125094), David E. Allen & Michael McAleer & Abhay K. Singh, 2015, Tinbergen Institute Discussion Paper
- [Predicting Economic Indicators from Web Text use karke Sentiment Composition](http://www.ijcce.org/index.php?m=content&c=index&a=show&catid=39&id=358), Abby Levenberg, et al, 2014
- [JP Morgan NLP research results](https://www.jpmorgan.com/global/research/machine-learning)

### The Naive Bayes classifier

The Naive Bayes algorithm hai very popular ke liye text classification because low computational cost aur memory requirements facilitate training on very large, high-dimensional datasets. Its predictive performance can compete ke saath more complex models, provide karta hai a good baseline, aur hai best known ke liye successful spam detection.

The model relies on Bayes theorem aur the assumption that the various features hain independent ka each other given the outcome class. mein other words, ke liye a given outcome, knowing the value ka one feature (e.g. the presence ka a token mein a document) does not provide any information about the value ka another feature.

### Code example: news article classification

hum start ke saath an illustration ka the Naive Bayes model to classify 2,225 BBC news articles that hum know belong to five different categories.

Notebook [text_classification](04_text_classification.ipynb) contain karta hai the relevant examples.

### Code examples: sentiment analysis

Sentiment analysis hai one ka the most popular use karta hai ka natural language processing aur machine learning ke liye trading because positive or negative perspectives on assets or other price drivers hain likely to impact returns. 

Generally, modeling approaches to sentiment analysis rely on dictionaries as the TextBlob library or models trained on outcomes ke liye a specific domain. The latter hai preferable because it permits more targeted labeling, e.g. by tying text features to subsequent price changes rather than indirect sentiment scores.

See [data](../data) directory ke liye instructions on obtaining the data.

#### Binary classification: twitter data

hum illustrate machine learning ke liye sentiment analysis use karke a [Twitter dataset](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) ke saath binary polarity labels, aur a large Yelp business review dataset ke saath a five-point outcome scale.

Notebook [sentiment_analysis_twitter](05_sentiment_analysis_twitter.ipynb) contain karta hai the relevant example.

- [Cheng-Caverlee-Lee September 2009 - January 2010 Twitter Scrape](https://archive.org/details/twitter_cikm_2010)

#### Comparing different ML algorithms on large, multiclass Yelp data

To illustrate text processing aur classification at larger scale, hum also use the [Yelp Dataset](https://www.yelp.com/dataset).

Notebook [sentiment_analysis_yelp](06_sentiment_analysis_yelp.ipynb) contain karta hai the relevant example.

- [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge)