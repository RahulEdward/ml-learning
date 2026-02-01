# Generative Adversarial Nets ke liye Synthetic Time Series Data

Yeh chapter introduces generative adversarial networks (GAN). GANs train a generator aur a discriminator network mein a competitive setting so that the generator learns to produce samples that the discriminator cannot distinguish from a given class ka training data. The goal hai to yield a generative model capable ka producing synthetic samples representative ka this class.
While most popular ke saath image data, GANs have also been used to generate synthetic time-series data mein the medical domain. Subsequent experiments ke saath financial data explored whether GANs can produce alternative price trajectories useful ke liye ML training or strategy backtests. hum replicate the 2019 NeurIPS Time-Series GAN paper to illustrate the approach aur demonstrate the results.

<p align="center">
<img src="https://i.imgur.com/W1Rp89K.png" width="60%">
</p>

## Vishay-suchi (Content)

1. [Generative adversarial networks ke liye synthetic data](#generative-adversarial-networks-ke liye-synthetic-data)
    * [Comparing generative and discriminative models](#comparing-generative-and-discriminative-models)
    * [Adversarial training: a zero-sum game of trickery](#adversarial-training-a-zero-sum-game-of-trickery)
2. [Code example: How to build a GAN use karke TensorFlow 2](#code-example-how-to-build-a-gan-use karke-tensorflow-2)
3. [Code example: TimeGAN: Adversarial Training ke liye Synthetic Financial Data](#code-example-timegan-adversarial-training-ke liye-synthetic-financial-data)
    * [Learning the data generation process across features and time](#learning-the-data-generation-process-across-features-and-time)
    * [Combining adversarial and supervised training with time-series embedding](#combining-adversarial-and-supervised-training-with-time-series-embedding)
    * [The four components of the TimeGAN architecture](#the-four-components-of-the-timegan-architecture)
    * [Implementing TimeGAN using TensorFlow 2](#implementing-timegan-using-tensorflow-2)
    * [Evaluating the quality of synthetic time-series data](#evaluating-the-quality-of-synthetic-time-series-data)
4. [Resources](#resources)
    * [How GAN's work](#how-gans-work)
    * [Implementation](#implementation)
    * [The rapid evolution of the GAN architecture zoo](#the-rapid-evolution-of-the-gan-architecture-zoo)
    * [Applications](#applications)

## Generative adversarial networks ke liye synthetic data

Yeh book mostly focuses on supervised learning algorithms that receive input data aur predict an outcome, which hum can compare to the ground truth to evaluate their performance. Such algorithms hain also called discriminative models because they learn to differentiate between different output values.
Generative adversarial networks (GANs) hain an instance ka generative models like the variational autoencoder hum encountered mein the [last chapter](../20_autoencoders_for_conditional_risk_factors).

### Comparing generative aur discriminative models

Discriminative models learn how to differentiate among outcomes y, given input data X. mein other words, they learn the probability ka the outcome given the data: p(y | X). Generative models, on the other hand, learn the joint distribution ka inputs aur outcome p(y, X). 

While generative models can be used as discriminative models use karke Bayes Rule to compute which class hai most likely (see [Chapter 10](../10_bayesian_machine_learning)), it appears often preferable to solve the prediction problem directly rather than by solving the more general generative challenge first.

### Adversarial training: a zero-sum game ka trickery

The key innovation ka GANs hai a new way ka learning the data-generating probability distribution. The algorithm sets up a competitive, or adversarial game between two neural networks called the generator aur the discriminator.

<p align="center">
<img src="https://i.imgur.com/0vuUsY0.png" width="80%">
</p>

## Code example: How to build a GAN use karke TensorFlow 2

To illustrate the implementation ka a generative adversarial network use karke Python, hum use the deep convolutional GAN (DCGAN) example discussed earlier mein this section to synthesize images from the fashion MNIST dataset that hum first encountered mein Chapter 13. 

Notebook [deep_convolutional_generative_adversarial_network](01_deep_convolutional_generative_adversarial_network.ipynb) illustrates the implementation ka a GAN use karke Python. It use karta hai the Deep Convolutional GAN (DCGAN) example to synthesize images from the fashion MNIST dataset

## Code example: TimeGAN: Adversarial Training ke liye Synthetic Financial Data

Generating synthetic time-series data poses specific challenges above aur beyond those encountered when designing GANs ke liye images. 
mein addition to the distribution over variables at any given point, such as pixel values or the prices ka numerous stocks, a generative model ke liye time-series data should also learn the temporal dynamics that shapes how one sequence ka observations follows another (see also discussion mein Chapter 9: [Time Series Models ke liye Volatility Forecasts aur Statistical Arbitrage](../09_time_series_models)).

Very recent aur promising [research](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks.pdf) by Yoon, Jarrett, aur van der Schaar, presented at NeurIPS mein December 2019, introduces a novel [Time-Series Generative Adversarial Network](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks.pdf) (TimeGAN) framework that aims to account ke liye temporal correlations by combining supervised aur unsupervised training. 
The model learns a time-series embedding space while optimizing both supervised aur adversarial objectives that encourage it to adhere to the dynamics observed while sampling from historical data during training. 
The authors test the model on various time series, including historical stock prices, aur find that the quality ka the synthetic data significantly outperforms that ka available alternatives.

### Learning the data generation process across features aur time

A successful generative model ke liye time-series data needs to capture both the cross-sectional distribution ka features at each point mein time aur the longitudinal relationships among these features over time. 
Expressed mein the image context hum just discussed, the model needs to learn not only what a realistic image looks like, but also how one image evolves from the next as mein a video.

### Combining adversarial aur supervised training ke saath time-series embedding

Prior attempts at generating time-series data like the recurrent (conditional) GAN relied on recurrent neural networks (RNN, see Chapter 19, [RNN ke liye Multivariate Time Series aur Sentiment Analysis](../19_recurrent_neural_nets)) mein the roles ka generator aur discriminator. 

TimeGAN explicitly incorporates the autoregressive nature ka time series by combining the unsupervised adversarial loss on both real aur synthetic sequences familiar from the DCGAN example ke saath a stepwise supervised loss ke saath respect to the original data. 
The goal hai to reward the model ke liye learning the distribution over transitions from one point mein time to the next present mein the historical data.

### The four components ka the TimeGAN architecture

The TimeGAN architecture combines an adversarial network ke saath an autoencoder aur has thus four network components as depicted mein Figure 21.4:
Autoencoder: embedding aur recovery networks
Adversarial Network: sequence generator aur sequence discriminator components
<p align="center">
<img src="https://i.imgur.com/WqoXbr8.png" width="80%">
</p>

### Implementing TimeGAN use karke TensorFlow 2

mein this section, hum implement the TimeGAN architecture just described. The authors provide sample code use karke TensorFlow 1 that hum port to TensorFlow 2. Building aur training TimeGAN requires several steps:
1. Selecting aur preparing real aur random time series inputs
2. Creating the key TimeGAN model components
3. Defining the various loss functions aur train steps used during the three training phases
4. Running the training loops aur logging the results
5. Generating synthetic time series aur evaluating the results

Notebook [TimeGAN_TF2](02_TimeGAN_TF2.ipynb) shows how to implement these steps.

### Evaluating the quality ka synthetic time-series data

The TimeGAN authors assess the quality ka the generated data ke saath respect to three practical criteria:
1. **Diversity**: the distribution ka the synthetic samples should roughly match that ka the real data
2. **Fidelity**: the sample series should be indistinguishable from the real data, aur 
3. **Usefulness**: the synthetic data should be as useful as their real counterparts ke liye solving a predictive task

The authors apply three methods to evaluate whether the synthetic data actually exhibits these characteristics:
1. **Visualization**: ke liye a qualitative diversity assessment ka diversity, hum use dimensionality reduction (principal components analysis (PCA) aur t-SNE, see Chapter 13) to visually inspect how closely the distribution ka the synthetic samples resembles that ka the original data
2. **Discriminative Score**: ke liye a quantitative assessment ka fidelity, the test error ka a time-series classifier such as a 2-layer LSTM (see Chapter 18) let’s us evaluate whether real aur synthetic time series can be differentiated or hain, mein fact, indistinguishable.
3. **Predictive Score**: ke liye a quantitative measure ka usefulness, hum can compare the test errors ka a sequence prediction model trained on, alternatively, real or synthetic data to predict the next time step ke liye the real data.

Notebook [evaluating_synthetic_data](03_evaluating_synthetic_data.ipynb) contain karta hai the relevant code samples.

## Sansadhan (Resources)

### How GAN's work

- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf), Ian Goodfellow, 2017
- [Why hai unsupervised learning important?](https://www.quora.com/Why-hai-unsupervised-learning-important), Yoshua Bengio on Quora, 2018
- [GAN Lab: Understanding Complex Deep Generative Models use karke Interactive Visual Experimentation](https://www.groundai.com/project/gan-lab-understanding-complex-deep-generative-models-use karke-interactive-visual-experimentation/), Minsuk Kahng, Nikhil Thorat, Duen Horng (Polo) Chau, Fernanda B. Viégas, aur Martin Wattenberg, IEEE Transactions on Visualization aur Computer Graphics, 25(1) (VAST 2018), Jan. 2019
    - [GitHub](https://poloclub.github.io/ganlab/)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), Ian Goodfellow, et al, 2014
- [Generative Adversarial Networks: an Overview](https://arxiv.org/pdf/1710.07035.pdf), Antonia Creswell, et al, 2017
- [Generative Models](https://blog.openai.com/generative-models/), OpenAI Blog

### Implementation

- [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)
- [CycleGAN](https://www.tensorflow.org/tutorials/generative/cyclegan)
- [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN), numerous Keras GAN implementations
- [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN), numerous PyTorch GAN implementations


### The rapid evolution ka the GAN architecture zoo

- [Unsupervised Representation Learning ke saath Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf), Luke Metz et al, 2016
- [Conditional Generative Adversarial Net](https://arxiv.org/pdf/1411.1784.pdf), Medhi Mirza aur Simon Osindero, 2014
- [Infogan: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf), Xi Chen et al, 2016
- [Stackgan: Text to Photo-realistic Image Synthesis ke saath Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242.pdf), Shaoting Zhang et al, 2016
- [Photo-realistic Single Image Super-resolution use karke a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf), Alejando Acosta et al, 2016
- [Unpaired Image-to-image Translation use karke Cycle-consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf), Juan-Yan Zhu et al, 2018
- [Learning What aur Where to Draw](https://arxiv.org/abs/1610.02454), Scott Reed, et al 2016
- [Fantastic GANs aur where to find them](http://guimperarnau.com/blog/2017/03/Fantastic-GANs-aur-where-to-find-them)

### Applications

- [Real-valued (Medical) Time Series Generation ke saath Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633), Cristóbal Esteban, Stephanie L. Hyland, Gunnar Rätsch, 2016
    - [GitHub Repo](https://github.com/ratschlab/RGAN)
- [MAD-GAN: Multivariate Anomaly Detection ke liye Time Series Data ke saath Generative Adversarial Networks](https://arxiv.org/pdf/1901.04997.pdf), Dan Li, Dacheng Chen, Jonathan Goh, aur See-Kiong Ng, 2019
    - [GitHub Repo](https://github.com/LiDan456/MAD-GANs)
- [GAN — Some cool applications](https://medium.com/@jonathan_hui/gan-some-cool-applications-ka-gans-4c9ecca35900), Jonathan Hui, 2018
- [gans-awesome-applications](https://github.com/nashory/gans-awesome-applications), curated list ka awesome GAN applications



