# RNN ke liye Trading: Multivariate Time Series aur Text Data

The major innovation ka RNN hai that each output hai a function ka both previous output aur new data. As a result, RNN gain the ability to incorporate information on previous observations into the computation it performs on a new feature vector, effectively creating a model ke saath memory. Yeh recurrent formulation enables parameter sharing across a much deeper computational graph that includes cycles. Prominent architectures include Long Short-Term Memory (LSTM) aur Gated Recurrent Units (GRU) that aim to overcome the challenge ka vanishing gradients associated ke saath learning long-range dependencies, where errors need to be propagated over many connections. 

RNNs have been successfully applied to various tasks that require mapping one or more input sequences to one or more output sequences aur hain particularly well suited to natural language. RNN can also be applied to univariate aur multivariate time series to predict market or fundamental data. Yeh chapter covers how RNN can model alternative text data use karke the word embeddings that hum covered mein [Chapter 16](16_word_embeddings) to classify the sentiment expressed mein documents.

## Vishay-suchi (Content)

1. [How recurrent neural nets work](#how-recurrent-neural-nets-work)
    * [Backpropagation through Time](#backpropagation-through-time)
    * [Alternative RNN Architectures](#alternative-rnn-architectures)
        - [Long-Short Term Memory](#long-short-term-memory)
        - [Gated Recurrent Units](#gated-recurrent-units)
2. [RNN ke liye financial time series ke saath TensorFlow 2](#rnn-ke liye-financial-time-series-ke saath-tensorflow-2)
    * [Code example: Univariate time-series regression: predicting the S&P 500](#code-example-univariate-time-series-regression-predicting-the-sp-500)
    * [Code example: Stacked LSTM for predicting weekly stock price moves and returns](#code-example-stacked-lstm-for-predicting-weekly-stock-price-moves-and-returns)
    * [Code example: Predicting returns instead of directional price moves](#code-example-predicting-returns-instead-of-directional-price-moves)
    * [Code example: Multivariate time-series regression for macro data](#code-example-multivariate-time-series-regression-for-macro-data)
3. [RNN ke liye text data: sentiment analysis aur return prediction](#rnn-ke liye-text-data-sentiment-analysis-aur-return-prediction)
    * [Code example: LSTM with custom word embeddings for sentiment classification](#code-example-lstm-with-custom-word-embeddings-for-sentiment-classification)
    * [Code example: Sentiment analysis with pretrained word vectors](#code-example-sentiment-analysis-with-pretrained-word-vectors)
    * [Code example: SEC filings for a bidirectional RNN GRU to predict weekly returns](#code-example-sec-filings-for-a-bidirectional-rnn-gru-to-predict-weekly-returns)

## How recurrent neural nets work

RNNs assume that the input data has been generated as a sequence such that previous data points impact the current observation aur hain relevant ke liye predicting subsequent values. Thus, they allow ke liye more complex input-output relationships than FFNNs aur CNNs, which hain designed to map one input vector to one output vector use karke a given number ka computational steps. 
RNNs, mein contrast, can model data ke liye tasks where the input, the output, or both, hain best represented as a sequence ka vectors. 

ke liye a thorough overview, see [chapter 10](https://www.deeplearningbook.org/contents/rnn.html mein [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow, Bengio, aur Courville (2016).

### Backpropagation through Time

 RNNs hain called recurrent because they apply the same transformations to every element ka a sequence mein a way that the output depends on the outcome ka prior iterations. As a result, RNNs maintain an internal state that captures information about previous elements mein the sequence akin to a memory.

The backpropagation algorithm that updates the weight parameters based on the gradient ka the loss function ke saath respect to the parameters involves a forward pass from left to right along the unrolled computational graph, followed by backward pass mein the opposite direction.

- [Sequence Modeling: Recurrent aur Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html), Deep Learning Book, Chapter 10, Ian Goodfellow, Yoshua Bengio aur Aaron Courville, MIT Press, 2016
- [Supervised Sequence Labelling ke saath Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/preprint.pdf), Alex Graves, 2013
- [Tutorial on LSTM Recurrent Networks](http://people.idsia.ch/~juergen/lstm/sld001.htm), Juergen Schmidhuber, 2003
- [The Unreasonable Effectiveness ka Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Alternative RNN Architectures

RNNs can be designed mein a variety ka ways to best capture the functional relationship aur dynamic between input aur output data. mein addition to the recurrent connections between the hidden states, there hain several alternative approaches, including recurrent output relationships, bidirectional RNN, aur encoder-decoder architectures.

#### Long-Short Term Memory

RNNs ke saath an LSTM architecture have more complex units that maintain an internal state aur contain gates to keep track ka dependencies between elements ka the input sequence aur regulate the cell’s state accordingly. These gates recurrently connect to each other instead ka the usual hidden units hum encountered above. They aim to address the problem ka vanishing aur exploding gradients by letting gradients pass through unchanged.

A typical LSTM unit combines four parameterized layers that interact ke saath each other aur the cell state by transforming aur passing along vectors. These layers usually involve an input gate, an output gate, aur a forget gate, but there hain variations that may have additional gates or lack some ka these mechanisms

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), Christopher Olah, 2015
- [An Empirical Exploration ka Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf), Rafal Jozefowicz, Ilya Sutskever, et al, 2015

#### Gated Recurrent Units

Gated recurrent units (GRU) simplify LSTM units by omitting the output gate. They have been shown to achieve similar performance on certain language modeling tasks but do better on smaller datasets.

- [Learning Phrase Representations use karke RNN Encoder–Decoder ke liye Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf), Kyunghyun Cho, Yoshua Bengio, et al 2014
- [Empirical Evaluation ka Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555), Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio, 2014

## RNN ke liye financial time series ke saath TensorFlow 2

hum illustrate how to build RNN use karke the Keras library ke liye various scenarios. The first set ka models includes regression aur classification ka univariate aur multivariate time series. The second set ka tasks focuses on text data ke liye sentiment analysis use karke text data converted to word embeddings (see [Chapter 15](../15_word_embeddings)). 

- [Recurrent Neural Networks (RNN) ke saath Keras](https://www.tensorflow.org/guide/keras/rnn)
- [Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Keras documentation](https://keras.io/getting-started/sequential-model-guide/)
- [LSTM documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Working ke saath RNNs](https://keras.io/guides/working_with_rnns/) by Scott Zhu aur Francois Chollet

### Code example: Univariate time-series regression: predicting the S&P 500

Notebook [univariate_time_series_regression](01_univariate_time_series_regression.ipynb) demonstrate karta hai how to get data into the requisite shape aur how to forecast the S&P 500 index values use karke a Recurrent Neural Network. 

### Code example: Stacked LSTM ke liye predicting weekly stock price moves aur returns

hum'll now build a slightly deeper model by stacking two LSTM layers use karke the Quandl stock price data. Furthermore, hum will include features that hain not sequential mein nature, namely indicator variables that identify the ticker aur time periods like month aur year.
- See the [stacked_lstm_with_feature_embeddings](02_stacked_lstm_with_feature_embeddings.ipynb) notebook ke liye implementation details.

### Code example: Predicting returns instead ka directional price moves

Notebook [stacked_lstm_with_feature_embeddings_regression](03_stacked_lstm_with_feature_embeddings_regression.ipynb) illustrates how to adapt the model to the regression task ka predicting returns rather than binary price changes.

### Code example: Multivariate time-series regression ke liye macro data

So far, hum have limited our modeling efforts to single time series. RNNs hain naturally well suited to multivariate time series aur represent a non-linear alternative to the Vector Autoregressive (VAR) models hum covered mein [Chapter 9, Time Series Models](../09_time_series_models).

Notebook [multivariate_timeseries](04_multivariate_timeseries.ipynb) demonstrate karta hai the application ka RNNs to modeling aur forecasting several time series use karke the same dataset hum used ke liye the [VAR example](../09_time_series_models/04_vector_autoregressive_model.ipynb), namely monthly data on consumer sentiment, aur industrial production from the Federal Reserve's FRED service.

## RNN ke liye text data: sentiment analysis aur return prediction

### Code example: LSTM ke saath custom word embeddings ke liye sentiment classification

RNNs hain commonly applied to various natural language processing tasks. hum've already encountered sentiment analysis use karke text data mein part three ka [this book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=VMKJPZC4N36TTZZCWATP&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=8f331266-0d21-4c76-a3eb-d2e61d23bb31&pd_rd_w=kVGNF&pd_rd_wg=LYLKH&ref_=pd_gw_ci_mcx_mr_hp_d).

Yeh example shows how to learn custom embedding vectors while training an RNN on the classification task. Yeh differs from the word2vec model that learns vectors while optimizing predictions ka neighboring tokens, resulting mein their ability to capture certain semantic relationships among words (see Chapter 16). Learning word vectors ke saath the goal ka predicting sentiment implies that embeddings will reflect how a token relates to the outcomes it hai associated ke saath.

Notebook [sentiment_analysis_imdb](05_sentiment_analysis_imdb.ipynb) illustrates how to apply an RNN model to text data to detect positive or negative sentiment (which can easily be extended to a finer-grained sentiment scale). hum hain going to use word embeddings to represent the tokens mein the documents. hum covered word embeddings mein [Chapter 15, Word Embeddings](../15_word_embeddings). They hain an excellent technique to convert text into a continuous vector representation such that the relative location ka words mein the latent space encodes useful semantic aspects based on the words' usage mein context.

### Code example: Sentiment analysis ke saath pretrained word vectors

mein [Chapter 15, Word Embeddings](../15_word_embeddings), hum showed how to learn domain-specific word embeddings. Word2vec, aur related learning algorithms, produce high-quality word vectors, but require large datasets. Hence, it hai common that research groups share word vectors trained on large datasets, similar to the weights ke liye pretrained deep learning models that hum encountered mein the section on transfer learning mein the [previous chapter](../17_convolutional_neural_nets).

Notebook [sentiment_analysis_pretrained_embeddings](06_sentiment_analysis_pretrained_embeddings.ipynb) illustrates how to use pretrained Global Vectors ke liye Word Representation (GloVe) provided by the Stanford NLP group ke saath the IMDB review dataset.

- [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), Stanford AI Group
- [GloVe: Global Vectors ke liye Word Representation](https://nlp.stanford.edu/projects/glove/), Stanford NLP

### Code example: SEC filings ke liye a bidirectional RNN GRU to predict weekly returns

mein Chapter 16, hum discussed important differences between product reviews aur financial text data. While the former was useful to illustrate important workflows, mein this section, hum will tackle more challenging but also more relevant financial documents. 

More specifically, hum will use the SEC filings data introduced mein [Chapter 16](../16_word_embeddings) to learn word embeddings tailored to predicting the return ka the ticker associated ke saath the disclosures from before publication to one week after.

Notebook [sec_filings_return_prediction](07_sec_filings_return_prediction.ipynb) contain karta hai the code examples ke liye this application. 

See the notebook [sec_preprocessing](../16_word_embeddings/06_sec_preprocessing.ipynb) mein Chapter 16 aur instructions mein the data folder on GitHub on how to obtain the data.
