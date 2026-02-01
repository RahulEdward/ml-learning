# Convolutional Neural Networks: Time Series as Images

mein this chapter, hum introduce the first specialized Deep Learning architectures that hum will cover mein part 4. Deep Convolutional Neural Networks, also ConvNets or CNN, have enabled superhuman performance mein classifying images, video, speech, aur audio. Recurrent nets, the subject ka the following chapter, have performed exceptionally well on sequential data such as text aur speech.

CNNs hain named after the linear algebra operation called convolution that replaces the general matrix multiplication typical ka feed-forward networks (discussed mein the last chapter on Deep Learning) mein at least one ka their layers. hum will discuss how convolutions work aur why they hain particularly useful to data ke saath a certain regular structure like images or time series.

Research into CNN architectures has proceeded very rapidly aur new architectures that improve benchmark performance continue to emerge. hum will describe a set ka building blocks that consistently appears mein successful applications aur illustrate their application to image data aur financial time series. hum will also demonstrate how transfer learning can speed up learning by use karke pre-trained weights ke liye some ka the CNN layers.

## Vishay-suchi (Content)

1. [How CNNs learn to model grid-like data](#how-cnns-learn-to-model-grid-like-data)
    * [Code example: From hand-coding to learning and synthesizing filters from data](#code-example-from-hand-coding-to-learning-and-synthesizing-filters-from-data)
    * [How the key elements of a convolutional layer operate](#how-the-key-elements-of-a-convolutional-layer-operate)
    * [Computer Vision Tasks](#computer-vision-tasks)
    * [The evolution of CNN architectures: key innovations](#the-evolution-of-cnn-architectures-key-innovations)
2. [CNN ke liye Images: From Satellite Data to Object Detection](#cnn-ke liye-images-from-satellite-data-to-object-detection)
    * [Code example: LeNet5: The first CNN with industrial applications](#code-example-lenet5-the-first-cnn-with-industrial-applications)
    * [Code example: AlexNet - reigniting deep learning research](#code-example-alexnet---reigniting-deep-learning-research)
    * [Code example: transfer learning with VGG16 in practice](#code-example-transfer-learning-with-vgg16-in-practice)
        - [How to extract bottleneck features](#how-to-extract-bottleneck-features)
        - [How to fine-tune a pre-trained model](#how-to-fine-tune-a-pre-trained-model)
    * [Code example: identifying land use with satellite images using transfer learning](#code-example-identifying-land-use-with-satellite-images-using-transfer-learning)
    * [Code example: object detection in practice with Google Street View House Numbers](#code-example-object-detection-in-practice-with-google-street-view-house-numbers)
        - [Preprocessing the source images](#preprocessing-the-source-images)
        - [Transfer learning with a custom final layer for multiple outputs](#transfer-learning-with-a-custom-final-layer-for-multiple-outputs)
3. [CNN ke liye time series data: predicting stock returns](#cnn-ke liye-time-series-data-predicting-stock-returns)
    * [Code example: building an autoregressive CNN with 1D convolutions](#code-example-building-an-autoregressive-cnn-with-1d-convolutions)
    * [Code example: CNN-TA - clustering financial time series in 2D image format](#code-example-cnn-ta---clustering-financial-time-series-in-2d-image-format)
        - [Creating the 2D time series of financial indicators](#creating-the-2d-time-series-of-financial-indicators)
        - [Select and cluster the most relevant features](#select-and-cluster-the-most-relevant-features)
        - [Create and train a convolutional neural network](#create-and-train-a-convolutional-neural-network)
        - [Backtesting a long-short trading strategy](#backtesting-a-long-short-trading-strategy)

## How CNNs learn to model grid-like data

CNNs hain conceptually similar to the feedforward NNs hum covered mein the previous chapter. They consist ka units that contain parameters called weights aur biases, aur the training process adjusts these parameters to optimize the network’s output ke liye a given input. Each unit applies its parameters to a linear operation on the input data or activations received from other units, possibly followed by a non-linear transformation. 

CNNs differ because they encode the assumption that the input has a structure most commonly found mein image data where pixels form a two-dimensional grid, typically ke saath several channels to represent the components ka the color signal, such as the red, green aur blue channels ka the RGB color model.

The most important element to encode the assumption ka a grid-like topology hai the convolution operation that gives CNNs their name, combined ke saath pooling. hum will see that the specific assumptions about the functional relationship between input aur output data implies that CNNs need far fewer parameters aur compute more efficiently.

### Code example: From hand-coding to learning aur synthesizing filters from data

ke liye image data, this local structure has traditionally motivated the development ka hand-coded filters that extract such patterns ke liye the use as features mein machine learning models.
- Notebook [filter_example](01_filter_example.ipynb) illustrates how to use hand-coded filters mein a convolutional network aur visualize the resulting transformation ka the image.
- See [Interpretability ka Deep Learning Models ke saath Tensorflow 2.0](https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow) ke liye an example visualization ka the patterns learned by CNN filters.

### How the key elements ka a convolutional layer operate

Fully-connected feedforwardNNs make no assumptions about the topology, or local structure ka the input data so that arbitrarily reordering the features has no impact on the training result.

ke liye many data sources, however, local structure hai quite significant. Examples include autocorrelation mein time series or the spatial correlation among pixel values due to common patterns like edges or corners. ke liye image data, this local structure has traditionally motivated the development ka hand-coded filter methods that extract local patterns ke liye the use as features mein machine learning models.

- [Deep Learning](http://www.deeplearningbook.org/contents/convnets.html), Chapter 9, Convolutional Networks, Ian Goodfellow et al, MIT Press, 2016
- [CS231n: Convolutional Neural Networks ke liye Visual Recognition](http://cs231n.stanford.edu/syllabus.html), Stanford’s deep learning course. Helpful ke liye building foundations, ke saath engaging lectures aur illustrative problem sets.
- [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/#conv), Module 2 mein CS231n Convolutional Neural Networks ke liye Visual Recognition, Lecture Notes by Andrew Karpathy, Stanford, 2016
- [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](http://www.image-net.org/challenges/LSVRC/)
- [Convnet Benchmarks](https://github.com/soumith/convnet-benchmarks), Benchmarking ka all publicly accessible implementations ka convnets
- [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html), ConvNetJS CIFAR-10 demo mein the browser by Andrew Karpathy
- [An Interactive Node-Link Visualization ka Convolutional Neural Networks](http://scs.ryerson.ca/~aharley/vis/), interactive CNN visualization
- [GradientBased Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf), Yann LeCun Leon Bottou Yoshua Bengio aur Patrick, IEEE, 1998
- [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/), Christopher Olah, 2014
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122), Fisher Yu, Vladlen Koltun, ICLR 2016

### Computer Vision Tasks

Image classification hai a fundamental computer vision task that requires labeling an image based on certain objects it contain karta hai. Many practical applications, including investment aur trading strategies, require additional information. 
- The object detection task requires not only the identification but also the spatial location ka all objects ka interest, typically use karke bounding boxes. Several algorithms have been developed to overcome the inefficiency ka brute-force sliding-window approaches, including region proposal methods (R-CNN) aur the You Only Look Once (YOLO) real-time object detection algorithm (see references on GitHub).
- The object segmentation task goes a step further aur requires a class label aur an outline ka every object mein the input image. Yeh may be useful to count objects mein an image aur evaluate a level ka activity. 
- Semantic segmentation, also called scene parsing, makes dense predictions to assign a class label to each pixel mein the image. As a result, the image hai divided into semantic regions aur each pixel hai assigned to its enclosing object or region.

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/), You Only Look Once real-time object detection
- [Rich feature hierarchies ke liye accurate object detection aur semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf), Girshick et al, Berkely, arxiv 2014
- [Playing around ke saath RCNN](https://cs.stanford.edu/people/karpathy/rcnn/), Andrew Karpathy, Stanford
- [R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e), Rohith Ghandi, 2018

### The evolution ka CNN architectures: key innovations

Several CNN architectures have pushed performance boundaries over the past two decades by introducing important innovations. Predictive performance growth accelerated dramatically ke saath the arrival ka big data mein the form ka ImageNet (Fei-Fei 2015) ke saath 14 million images assigned to 20,000 classes by humans via Amazon’s Mechanical Turk. The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) became the focal point ka CNN progress around a slightly smaller set ka 1.2 million images from 1,000 classes.

- [Fully Convolutional Networks ke liye Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf), Long et al, Berkeley
- [Mask R-CNN](https://arxiv.org/abs/1703.06870), Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, arxiv, 2017
- [U-Net: Convolutional Networks ke liye Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf), Olaf Ronneberger, Philipp Fischer, aur Thomas Brox, arxiv 2015
- [U-Net Tutorial](http://deeplearning.net/tutorial/unet.html)
- [Very Deep Convolutional Networks ke liye Large-Scale Visual Recognition](http://www.robots.ox.ac.uk/~vgg/research/very_deep/), Karen Simonyan aur Andrew Zisserman on VGG16 that won the ImageNet ILSVRC-2014 competition
- [Benchmarks ke liye popular CNN models](https://github.com/jcjohnson/cnn-benchmarks)
- [Analysis ka deep neural networks](https://medium.com/@culurciello/analysis-ka-deep-neural-networks-dcf398e71aae), Alfredo Canziani, Thomas Molnar, Lukasz Burzawa, Dawood Sheik, Abhishek Chaurasia, Eugenio Culurciello, 2018
- [LeNet-5 Demos](http://yann.lecun.com/exdb/lenet/index.html)
- [Neural Network Architectures](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba)
- [Deep Residual Learning ke liye Image Recognition](https://arxiv.org/pdf/1512.03385.pdf), Kaiming He et al, Microsoft Research, 2015
- [Rethinking the Inception Architecture ke liye Computer Vision](https://arxiv.org/abs/1512.00567), Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, arxiv 2015
- [Inception-v4, Inception-ResNet aur the Impact ka Residual Connections on Learning](https://arxiv.org/abs/1602.07261), Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, arxiv, 2016
- [Network mein Network](https://arxiv.org/pdf/1312.4400v3.pdf), Min Lin et al, arxiv 2014
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), Sergey Ioffe, Christian Szegedy, arxiv 2015
- [An Overview ka ResNet aur its Variants](https://towardsdatascience.com/an-overview-ka-resnet-aur-its-variants-5281e2f56035), Vincent Fung, 2017

## CNN ke liye Images: From Satellite Data to Object Detection

Yeh section demonstrate karta hai how to solve key computer vision tasks such as image classification aur object detection. As mentioned mein the introduction aur mein Chapter 3 on alternative data, image data can inform a trading strategy by providing clues about future trends, changing fundamentals, or specific events relevant ke liye a target asset class or investment universe. Popular examples include exploiting satellite images ke liye clues about the supply ka agricultural commodities, consumer aur economic activity, or the status ka manufacturing or raw material supply chains. Specific tasks might include, ke liye example: 
- Image classification: identify whether cultivated land ke liye certain crops hai expanding or predict harvest quality aur quantities, or 
- Object detection: count the number ka oil tankers on a certain transport route or the number ka cars mein a parking lot, or identify the location ka shoppers mein a mall.

### Code example: LeNet5: The first CNN ke saath industrial applications

All libraries hum introduced mein the last chapter provide support ke liye convolutional layers. 

Notebook [digit_classification_with_lenet5](02_digit_classification_with_lenet5.ipynb) illustrates the LeNet5 architecture use karke the most basic MNIST handwritten digit dataset,

### Code example: AlexNet - reigniting deep learning research

Fast-forward to 2012, aur hum move on to the deeper aur more modern AlexNet architecture. hum will use the CIFAR10 dataset that use karta hai 60,000 ImageNet samples, compressed to 32x32 pixel resolution (from the original 224x224), but still ke saath three color channels. There hain only 10 ka the original 1,000 classes. 

See the notebook [image_classification_with_alexnet](03_image_classification_with_alexnet.ipynb) ke liye implementation, including the use ka data augmentation.

### Code example: transfer learning ke saath VGG16 mein practice

mein practice, hum often do not have enough data to train a CNN from scratch ke saath random initialization. Transfer learning hai a machine learning technique that repurposes a model trained on one set ka data ke liye another task. Naturally, it works if the learning from the first task carries over to the task ka interest. If successful, it can lead to better performance aur faster training that requires less labeled data than training a neural network from scratch on the target task.

Tensorflow 2, ke liye example, contain karta hai pre-trained models ke liye several ka the reference architectures discussed previously, namely VGG16 aur its larger version VGG19, ResNet50, InceptionV3, aur InceptionResNetV2, as well as MobileNet, DenseNet, NASNet, aur MobileNetV2.

The transfer learning approach to CNN relies on pre-training on a very large dataset like ImageNet. The goal hai that the convolutional filters extract a feature representation that generalizes to new images. mein a second step, it leverages the result to either initialize aur retrain a new CNN or as inputs to mein a new network that tackles the task ka interest.

CNN architectures typically use a sequence ka convolutional layers to detect hierarchical patterns, adding one or more fully-connected layers to map the convolutional activations to the outcome classes or values. The output ka the last convolutional layer that feeds into the fully-connected part hai called bottleneck features. hum can use the bottleneck features ka a pre-trained network as inputs into a new fully-connected network, usually after applying a ReLU activation function. 

mein other words, hum freeze the convolutional layers aur replace the dense part ka the network. An additional benefit hai that hum can then use inputs ka different sizes because it hai the dense layers that constrain the input size. 

Alternatively, hum can use the bottleneck features as inputs into a different machine learning algorithm. mein the AlexNet architecture, e.g., the bottleneck layer computes a vector ke saath 4096 entries ke liye each 224 x 224 input image. hum then use this vector as features ke liye a new model.

Alternatively, hum can go a step further aur not only replace aur retrain the classifier on top ka the CNN use karke new data but to also fine-tune the weights ka the pre-trained CNN. To achieve this, hum continue training, either only ke liye later layers while freezing the weights ka some earlier layers. The motivation hai to preserve presumably more generic patterns learned by lower layers, such as edge or color blob detectors while allowing later layers ka the CNN to adapt to the details ka a new task. ImageNet, e.g., contain karta hai a wide variety ka dog breeds which may lead to feature representations specifically useful ke liye differentiating between these classes.

- [Building powerful image classification models use karke very little data](https://blog.keras.io/building-powerful-image-classification-models-use karke-very-little-data.html)
- [How transferable hain features mein deep neural networks?](https://papers.nips.cc/paper/5347-how-transferable-hain-features-mein-deep-neural-networks.pdf), Jason Yosinski, Jeff Clune, Yoshua Bengio, aur Hod Lipson, NIPS, 2014
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

#### How to extract bottleneck features

Notebook [bottleneck_features](09_bottleneck_features.ipynb) illustrates how to download the pre-trained VGG16 model, either ke saath the final layers to generate predictions or without the final layers to extract the outputs produced by the bottleneck features.

#### How to fine-tune a pre-trained model

Notebook [transfer_learning](10_transfer_learning.ipynb), adapted from a TensorFlow 2 tutorial, demonstrate karta hai how to freeze some or all ka the layers ka a pre-trained model aur continue training use karke a new fully-connected set ka layers aur data ke saath a different format.

### Code example: identifying land use ke saath satellite images use karke transfer learning

Satellite images figure prominently among alternative data (see [Chapter 3](../03_alternative_data)). ke liye instance, commodity traders may rely on satellite images to predict the supply ka certain crops or activity at mining sites, oil or tanker traffic. 

To illustrate working ke saath this type ka data, hum load the [EuroSat dataset](https://arxiv.org/abs/1709.00029) included mein the TensorFlow 2 datasets (Helber et al. 2017). The EuroSat dataset includes around 27,000 images mein 64x64 format that represent 10 different types ka land use karta hai.
 
Notebook [satellite_images](11_satellite_images.ipynb) downloads the [DenseNet201](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201) architecture from `tensorflow.keras.applications` aur replace its final layers.

hum use 10 percent ka the training images ke liye validation purposes aur achieve the best out-ka-sample classification accuracy ka 97.96 percent after ten epochs. Yeh exceeds the performance cited mein the original paper ke liye the best performing ResNet-50 architecture ke saath 90-10 split.

### Code example: object detection mein practice ke saath Google Street View House Numbers

Object detection requires the ability to distinguish between several classes ka objects aur to decide how many aur which ka these objects hain present mein an image.

A prominent example hai Ian Goodfellow’s identification ka house numbers from Google’s street view dataset. It requires to identify 
- how many ka up to five digits make up the house number, 
- The correct digit ke liye each component, aur
- The proper order ka the constituent digits.

See the [data](../data) directory ke liye instructions on obtaining the dataset.

#### Preprocessing the source images

Notebooks [svhn_preprocessing](12_svhn_preprocessing.ipynb) contain karta hai code to produce a simplified, cropped dataset that use karta hai bounding box information to create regularly shaped 32x32 images containing the digits; the original images hain ka arbitrary shape.

#### Transfer learning ke saath a custom final layer ke liye multiple outputs

Notebook [svhn_object_detection](13_svhn_object_detection.ipynb) goes on to illustrate how to build a deep CNN use karke Keras’ functional API to generate multiple outputs: one to predict how many digits hain present, aur five ke liye the value ka each mein the order they appear.

## CNN ke liye time series data: predicting stock returns

CNN were originally developed to process image data aur have achieved superhuman performance on various computer vision tasks. As discussed mein the first section, time series data has a grid-like structure similar to that ka images, aur CNN have been successfully applied to one-, two- aur three dimensional representations ka temporal data. 

The application ka CNN to time series will most likely bear fruit if the data meets the model’s key assumption that local patterns or relationships help predict the outcome. mein the time-series context, local patterns could be autocorrelation or similar non-linear relationships at relevant intervals. Along the second aur third dimension, local patterns imply systematic relationships among different components ka a multivariate series or among these series ke liye different tickers. Since locality matters, it hai important that the data hai organized accordingly mein contrast to feed-forward networks where shuffling the elements ka any dimension does not negatively affect the learning process.

### Code example: building an autoregressive CNN ke saath 1D convolutions

hum will introduce the time series use case ke liye CNN ke saath a univariate autoregressive asset return model. More specifically, the model receives the most recent 12 months ka returns aur use karta hai a single layer ka one-dimensional convolutions to predict the subsequent month.

Notebook [time_series_prediction](04_time_series_prediction.ipynb) illustrates the time series use case ke saath the univariate asset price forecast example hum introduced mein the last chapter. Recall that hum create rolling monthly stock returns aur use the 24 lagged returns alongside one-hot-encoded month information to predict whether the subsequent monthly return hai positive or negative.

### Code example: CNN-TA - clustering financial time series mein 2D image format

To exploit the grid-like structure ka time-series data, hum can use CNN architectures ke liye univariate aur multivariate time series. mein the latter case, hum consider different time series as channels, similar to the different color signals.

An alternative approach converts a time series ka alpha factors into a two-dimensional format to leverage the ability ka CNNs to detect local patterns. [Sezer aur Ozbayoglu](https://www.sciencedirect.com/science/article/abs/pii/S1568494618302151) (2018) propose [CNN-TA](https://github.com/omerbsezer/CNN-TA) that computes 15 technical indicators ke liye different intervals aur use karta hai hierarchical clustering (see Chapter 13) to locate indicators that behave similarly close to each other mein a 2D grid.

#### Creating the 2D time series ka financial indicators

Notebook [engineer_cnn_features](05_cnn_for_trading_feature_engineering.ipynb) create karta hai technical indicators at different intervals.

#### Select aur cluster the most relevant features
 
Notebook [convert_cnn_features_to_image_format](06_cnn_for_trading_features_to_clustered_image_format.ipynb) selects the 15 most relevant features from the 20 candidates to fill the 15⨉15 input grid aur then applies hierarchical clustering.

#### Create aur train a convolutional neural network

Now hum hain ready to design, train aur evaluate a CNN following the steps outlined mein the previous section. Notebook [cnn_for_trading](07_cnn_for_trading.ipynb) contain karta hai the relevant code examples.

#### Backtesting a long-short trading strategy

To get a sense ka the signal quality, hum compute the spread between equal-weighted portfolios invested mein stocks selected according to the signal quintiles use karke [Alphalens](https://github.com/quantopian/alphalens) (see [Chapter 4](../04_alpha_factor_research)).

<p align="center">
<img src="https://i.imgur.com/JlKttDL.png" width="80%">
</p>




