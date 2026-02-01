# Autoencoders ke liye Conditional Risk Factors aur Asset Pricing

Yeh chapter shows how unsupervised learning can leverage deep learning ke liye trading. More specifically, hum’ll discuss autoencoders that have been around ke liye decades but recently attracted fresh interest.

An autoencoder hai a neural network trained to reproduce the input while learning a new representation ka the data, encoded by the parameters ka a hidden layer. 
Autoencoders have long been used ke liye nonlinear dimensionality reduction aur manifold learning (see [Chapter 13](../13_unsupervised_learning)). 
A variety ka designs leverage the feedforward, convolutional, aur recurrent network architectures hum covered mein the last three chapters. 

hum will also see how autoencoders can underpin a trading strategy by building a deep neural network that use karta hai an [autoencoder to extract risk factors](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) aur predict equity returns, conditioned on a range ka equity attributes (Gu, Kelly, aur Xiu 2020).

## Vishay-suchi (Content)

1. [Autoencoders ke liye nonlinear feature extraction](#autoencoders-ke liye-nonlinear-feature-extraction)
    * [Code example: Generalizing PCA with nonlinear dimensionality reduction](#code-example-generalizing-pca-with-nonlinear-dimensionality-reduction)
    * [Code example: convolutional autoencoders to compress and denoise images](#code-example-convolutional-autoencoders-to-compress-and-denoise-images)
    * [Seq2seq autoencoders to extract time-series features for trading](#seq2seq-autoencoders-to-extract-time-series-features-for-trading)
    * [Code example: Variational autoencoders - learning how to generate the input data](#code-example-variational-autoencoders---learning-how-to-generate-the-input-data)
2. [Code example: A conditional autoencoder ke liye return forecasts aur trading](#code-example-a-conditional-autoencoder-ke liye-return-forecasts-aur-trading)
    * [Creating a new dataset with stock price and metadata information](#creating-a-new-dataset-with-stock-price-and-metadata-information)
    * [Computing predictive asset characteristics](#computing-predictive-asset-characteristics)
    * [Creating and training the conditional autoencoder architecture](#creating-and-training-the-conditional-autoencoder-architecture)
    * [Evaluating the results](#evaluating-the-results)

## Autoencoders ke liye nonlinear feature extraction

mein Chapter 17, [Deep Learning ke liye Trading](../17_deep_learning), hum saw how neural networks succeed at supervised learning by extracting a hierarchical feature representation useful ke liye the given task. Convolutional neural networks, e.g., learn aur synthesize increasingly complex patterns from grid-like data, ke liye example, to identify or detect objects mein an image or to classify time series. 
An autoencoder, mein contrast, hai a neural network designed exclusively to learn a new representation that encodes the input mein a way that helps solve another task. To this end, the training forces the network to reproduce the input. Since autoencoders typically use the same data as input aur output, they hain also considered an instance ka self-supervised learning. 
mein the process, the parameters ka a hidden layer h become the code that represents the input, similar to the word2vec model covered mein [Chapter 16](../16_word_embeddings). 

ke liye a good overview, see Chapter 14 mein Deep Learning:
- [Autoencoders](http://www.deeplearningbook.org/contents/autoencoders.html), Ian Goodfellow, Yoshua Bengio aur Aaron Courville, Deep Learning Book, MIT Press 2016

The TensorFlow's Keras interface makes it fairly straightforward to build various types ka autoencoders aur the following examples hain adapted from Keras' tutorials.

- [Building Autoencoders mein Keras](https://blog.keras.io/building-autoencoders-mein-keras.html)

### Code example: Generalizing PCA ke saath nonlinear dimensionality reduction

A traditional use case includes dimensionality reduction, achieved by limiting the size ka the hidden layer so that it performs lossy compression. Such an autoencoder hai called undercomplete aur the purpose hai to force it to learn the most salient properties ka the data by minimizing a loss function. mein addition to feedforward architectures, autoencoders can also use convolutional layers to learn hierarchical feature representations.

Notebook [deep_autoencoders](01_deep_autoencoders.ipynb) illustrates how to implement several ka autoencoder models use karke TensorFlow, including autoencoders use karke deep feedforward nets aur sparsity constraints. 
 
### Code example: convolutional autoencoders to compress aur denoise images

As discussed mein Chapter 18, [CNNs: Time Series as Images aur Satellite Image Classification](../18_convolutional_neural_nets), fully-connected feedforward architectures hain not well suited to capture local correlations typical to data ke saath a grid-like structure. Instead, autoencoders can also use convolutional layers to learn a hierarchical feature representation. Convolutional autoencoders leverage convolutions aur parameter sharing to learn hierarchical patterns aur features irrespective ka their location, translation, or changes mein size.

Notebook [convolutional_denoising_autoencoders](02_convolutional_denoising_autoencoders.ipynb) goes on to demonstrate how to implement convolutional aur denoising autoencoders to recover corrupted image inputs.

### Seq2seq autoencoders to extract time-series features ke liye trading

Sequence-to-sequence autoencoders hain based on RNN components, such as Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRUs). They learn a compressed representation ka sequential data aur have been applied to video, text, audio, aur time-series data.

- [A ten-minute introduction to sequence-to-sequence learning mein Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-mein-keras.html), Francois Chollet, September 2017
- [Unsupervised Learning ka Video Representations use karke LSTMs](https://arxiv.org/abs/1502.04681), Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov, 2016
- [Gradient Trader Part 1: The Surprising Usefulness ka Autoencoders](https://rickyhan.com/jekyll/update/2017/09/14/autoencoders.html)
    - [Code examples](https://github.com/0b01/recurrent-autoencoder)
- [Deep Learning Financial Market Data](http://wp.doc.ic.ac.uk/hipeds/wp-content/uploads/sites/78/2017/01/Steven_Hutt_Deep_Networks_Financial.pdf)
    - Motivation: Regulators identify prohibited patterns of trading activity detrimental to orderly markets. Financial Exchanges are responsible for maintaining orderly markets. (e.g. Flash Crash and Hound of Hounslow.)
    - Challenge: Identify prohibited trading patterns quickly and efficiently.
    - **Goal**: Build a trading pattern search function using Deep Learning. Given a sample trading pattern identify similar patterns in historical LOB data.

### Code example: Variational autoencoders - learning how to generate the input data

Variational Autoencoders (VAE) hain more recent developments focused on generative modeling. More specifically, VAEs hain designed to learn a latent variable model ke liye the input data. Note that hum encountered latent variables mein Chapter 14, Topic Modeling.

Hence, VAEs do not let the network learn arbitrary functions as long as it faithfully reproduces the input. Instead, they aim to learn the parameters ka a probability distribution that generates the input data. mein other words, VAEs hain generative models because, if successful, you can generate new data points by sampling from the distribution learned by the VAE.

Notebook [variational_autoencoder](03_variational_autoencoder.ipynb) shows how to build a Variational Autoencoder use karke Keras.

- [Auto-encoding variational bayes](https://arxiv.org/abs/1312.6114), Diederik P Kingma, Max Welling, 2014
- [Tutorial: What hai a variational autoencoder?](https://jaan.io/what-hai-variational-autoencoder-vae-tutorial/)
- [Variational Autoencoder / Deep Latent Gaussian Model mein tensorflow aur pytorch](https://github.com/altosaar/variational-autoencoder)

## Code example: A conditional autoencoder ke liye return forecasts aur trading

Recent research by [Gu, Kelly, aur Xiu](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) develops an asset pricing model based on the exposure ka securities to risk factors. It builds on the concept ka data-driven risk factors that hum discussed mein Chapter 13 when introducing PCA as well as the risk factor models covered mein Chapter 4, Financial Feature Engineering: How to Research Alpha Factors. 
The authors aim to show that the asset characteristics used by factor models to capture the systematic drivers ka ‘anomalies’ hain just proxies ke liye the time-varying exposure to risk factors that cannot be directly measured. 
mein this context, anomalies hain returns mein excess ka those explained by the exposure to aggregate market risk (see the discussion ka the capital asset pricing model mein [Chapter 5](../05_strategy_evaluation)).

### Creating a new dataset ke saath stock price aur metadata information

The reference implementation use karta hai stock price aur firm characteristic data ke liye over 30,000 US equities from the Center ke liye Research mein Security Prices (CRSP) from 1957-2016 at monthly frequency. It computes 94 metrics that include a broad range ka asset attributes suggested as predictive ka returns mein previous academic research aur listed mein Green, Hand, aur Zhang (2017), who set out to verify these claims.
Since hum do not have access to the high-quality but costly CRSP data, hum leverage [yfinance](https://github.com/ranaroussi/yfinance) (see Chapter 2, [Market aur Fundamental Data: Sources aur Techniques](../02_market_and_fundamental_data)) to download price aur metadata from Yahoo Finance. There hain downsides to choosing free data, including:
- the lack ka quality control regarding adjustments, 
- survivorship bias because hum cannot get data ke liye stocks that hain no longer listed, aur
- a smaller scope mein terms ka both the number ka equities aur the length ka their history. 

Notebook [build_us_stock_dataset](04_build_us_stock_dataset.ipynb) contain karta hai the relevant code examples ke liye this section.

### Computing predictive asset characteristics

The authors test 94 asset attributes aur identify the 20 most influential metrics while asserting that feature importance drops off quickly thereafter. The top 20 stock characteristics fall into three categories, namely:
- Price trend, including (industry) momentum, short- aur long-term reversal, or the recent maximum return
- Liquidity such as turnover, dollar volume, or market capitalization
- Risk measures, ke liye instance, total aur idiosyncratic return volatility or market beta

ka these 20, hum limit the analysis to 16 ke liye which hum have or can approximate the relevant inputs. Notebook [conditional_autoencoder_for_trading_data](05_conditional_autoencoder_for_trading_data.ipynb) demonstrate karta hai how to calculate the relevant metrics.

### Creating aur training the conditional autoencoder architecture

The conditional autoencoder proposed by the authors allows ke liye time-varying return distributions that take into account changing asset characteristics. 
To this end, they extend standard autoencoder architectures that hum discussed mein the first section ka this chapter to allow ke liye features to shape the encoding.

Notebook [conditional_autoencoder_for_asset_pricing_model](06_conditional_autoencoder_for_asset_pricing_model.ipynb) demonstrate karta hai how to create aur train this architecture.

### Evaluating the results

Notebook [alphalens_analysis](07_alphalens_analysis.ipynb) measures the financial performance ka the model's prediction.


