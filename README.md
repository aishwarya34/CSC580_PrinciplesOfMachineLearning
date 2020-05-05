<div align="center">
    <h1 style="font-size:300%;">Gradually Updated Neural Networks for Large-Scale Image Recognition</h1>
    <h3 style="font-size:100%;">Aishwarya Radhakrishnan <br>
    March 14, 2020</h3>
</div>
  
<hr>



## Abstract / Introduction

*Convolutional Neural Networks(CNNs) decreases the computation cost for Computer Vision problems than the Deep Neural Networks(DNNs). Increasing the depth leads to increase in parameters of the CNN and hence, increase in computation cost. But as depth plays a vital role in Computer Vision problems, increasing the depth without increasing the computation cost can lead to increased learning. This is achieved by introducing
computation orderings to the channels within convolutional layers.*

*Gradually Updated Neural Networks (GUNN) as opposed to the default Simultaneously Updated Convolutional Network (SUNN / CNN), gradually updates the feature representations against the
traditional convolutional network that computes its output
simultaneously.  This is achieved by updating one channel at a time and using the newly computed parameter of this channel and old parameter of other channels to get another channels in a single convolutional layer. This is repeated for all the channels untill all the old parameters of a single convolutional layer are updated to new values. Thus a single convolutional layer is broken down into multiple gradually updated convolutional layer.*



## Prerequisites

Installing pip, tensorflow, keras, numpy <br>

Commands: <br>
sudo yum install python-pip <br>
sudo pip install keras <br>
sudo pip install tensorflow <br>
sudo pip install numpy <br>

## CIFAR-10 Dataset

We do not need to download CIFAR-10 dataset explicitly as we Tensorflow provides *cifar10.load_data()* API which automatically downloads and loads Training and Test set in numpy ndarray.

After loading the CIFAR-10 dataset we choose a subset of dataset as follows:

number of training examples = 15000 <br>
number of test examples = 3000 <br>
X_train shape: (15000, 32, 32, 3) <br>
Y_train shape: (15000, 3) <br>
X_test shape: (3000, 32, 32, 3) <br>
Y_test shape: (3000, 3) <br>


<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/CIFAR10.png' /><br>
<figcaption>Figure 1: CIFAR 10 dataset with 10 classes that are randomly chosen</figcaption></div>
</figure>



## GUNN Layer implementation (Keras Custom Layer)

Keras Custom Layer needs to be used because GUNN layer consist of convolutional network which decomposes
the original computation into multiple sequential channel-wise convolutional operations as opposed to single convolutional operations in Conv2D.


<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/SUNNvsGUNN.png' /><br>
<figcaption>Figure 2: SUNN vs GUNN. SUNN architecture is our usually used Convolution Layer while GUNN is the channel-wise decomposed architecture.</figcaption></div>
</figure>


The expansion rate is used to know how many channels are simultaneously updated when decomposing the channels. The kernel channel is the same as the expansion rate. The kernel size is given by fxf.

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GunnLayer.png' /><br>
<figcaption>Figure 3: Gunn2D Keras custom layer. The Gunn2D layer expands channel at K rate and also uses Residual Network identity block implementation. </figcaption></div>
</figure>

<br>

The Keras custom layer needs forward propagation definition only.  Since our custom layer has trainable weights, we need to use stateful custom operations using custom layer class definition. 

Note: If you want to build a keras model with a custom layer that performs a custom operation and has a custom gradient, you should use @tf.custom_gradient.


## Building GUNN-15 Model in Keras for 10 classes of CIFAR-10 dataset

Using Keras Convolutional Neural Networks building blocks and custom implemented Gunn2D layer created GUNN-15 Model.

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GunnModel.png' /><br>
<figcaption>Figure 4: Gunn Model. The Gunn Model has Gunn2D layer implemented at the layers marked as * i.e. Conv2*, Conv3* and Conv4* layers. </figcaption></div>
</figure>




Built Gradually updated Convolutional Neural Net following GUNN-15 model having 15 layers that classifies CIFAR-10's 3 classes in Colab with the following architecture:

- Training examples : 5000  
- Testing examples : 100  
- Batch Size : 50 
- epochs : 50
- Adam optimizer : for adaptive estimates of lower-order moments
- Loss function : Categorical crossentropy for multiple class estimation
- Activation layer : RELU
- Batch Normalization : for the network to use higher learning rate without vanishing or exploding gradients. 

In addition, all the layers parameters are defined in accordance with the model in figure 4.

I have also used Residual Network's Identity block for each Gunn2D layer where the state of layer before doing Gunn2D operations is added to the layer after performing Gunn2D operations. Residual network helps in learning identity function when the gradients vanishes which happend in deep networks. 



## Training

<figure>
<div align="center">
<img src='https://github.com/aishwarya34/CSC580_PrinciplesOfMachineLearning/blob/master/img/GUNN-15_model.png' /><br>
<figcaption>Figure 5: GUNN-15 model having custon Gunn2D layer. </figcaption></div>
</figure>

<br>

The above figure shows the High-Level Keras Automatic Differentiation graph having different layers like Conv2d, Activation etc that are Keras APIs as well as the **Gunn2D layer** which is a custom layer.



Trained the Convolutional Neural Net to classify CIFAR-10's 3 classes in Colab using 5000 training examples and 100 testing examples with batch size of 50 and for 50 epochs, I get the following  output.


Epoch 1/100 <br>
50/50 [==============================] - 1432s 29s/step - loss: 2.2360 - categorical_accuracy: 0.3160 <br>
Epoch 2/100 <br>
50/50 [==============================] - 1398s 28s/step - loss: 2.1116 - categorical_accuracy: 0.3550 <br>
Epoch 3/100 <br>
50/50 [==============================] - 1388s 28s/step - loss: 1.9972 - categorical_accuracy: 0.3550 <br>
Epoch 4/100 <br>
50/50 [==============================] - 1398s 28s/step - loss: 1.8943 - categorical_accuracy: 0.3550 <br>
Epoch 5/100 <br>
50/50 [==============================] - 1393s 28s/step - loss: 1.8017 - categorical_accuracy: 0.3550 <br>
Epoch 6/100 <br>
50/50 [==============================] - 1384s 28s/step - loss: 1.7186 - categorical_accuracy: 0.3550 <br>
Epoch 7/50 <br>
50/50 [==============================] - 1377s 28s/step - loss: 1.6449 - categorical_accuracy: 0.3550 <br>
Epoch 8/100 <br>
50/50 [==============================] - 1373s 27s/step - loss: 1.5792 - categorical_accuracy: 0.3550 <br>
Epoch 9/100 <br>
50/50 [==============================] - 1385s 28s/step - loss: 1.5204 - categorical_accuracy: 0.3550 <br>
Epoch 10/100 <br>
50/50 [==============================] - 1388s 28s/step - loss: 1.4700 - categorical_accuracy: 0.3550 <br>
Epoch 11/100 <br>
50/50 [==============================] - 1381s 28s/step - loss: 1.4231 - categorical_accuracy: 0.3550 <br>
Epoch 12/100 <br>
50/50 [==============================] - 1384s 28s/step - loss: 1.3849 - categorical_accuracy: 0.3550 <br>
Epoch 13/100 <br>
50/50 [==============================] - 1382s 28s/step - loss: 1.3457 - categorical_accuracy: 0.3550 <br>
Epoch 14/100 <br>
50/50 [==============================] - 1403s 28s/step - loss: 1.3118 - categorical_accuracy: 0.4270 <br>
Epoch 15/100 <br>
50/50 [==============================] - 1398s 28s/step - loss: 1.2808 - categorical_accuracy: 0.5560 <br>
Epoch 16/100 <br>
50/50 [==============================] - 1384s 28s/step - loss: 1.2505 - categorical_accuracy: 0.5540 <br>
Epoch 17/100 <br>
50/50 [==============================] - 1390s 28s/step - loss: 1.2204 - categorical_accuracy: 0.5610 <br>
Epoch 18/100 <br>
50/50 [==============================] - 1376s 28s/step - loss: 1.1898 - categorical_accuracy: 0.5550 <br>
Epoch 19/100 <br>
50/50 [==============================] - 1385s 28s/step - loss: 1.1730 - categorical_accuracy: 0.5250 <br>
Epoch 20/100 <br>
50/50 [==============================] - 1389s 28s/step - loss: 1.1443 - categorical_accuracy: 0.5350 <br>
Epoch 21/100 <br>
50/50 [==============================] - 1380s 28s/step - loss: 1.0989 - categorical_accuracy: 0.5760 <br>
Epoch 22/100 <br>
50/50 [==============================] - 1390s 28s/step - loss: 1.0757 - categorical_accuracy: 0.5610 <br>
Epoch 23/100 <br>
50/50 [==============================] - 1383s 28s/step - loss: 1.0449 - categorical_accuracy: 0.5810 <br>
Epoch 24/100 <br>
50/50 [==============================] - 1393s 28s/step - loss: 1.0258 - categorical_accuracy: 0.5790 <br>
Epoch 25/100 <br>
50/50 [==============================] - 1386s 28s/step - loss: 1.0040 - categorical_accuracy: 0.5730 <br>
Epoch 26/100 <br>
50/50 [==============================] - 1378s 28s/step - loss: 0.9987 - categorical_accuracy: 0.5620 <br>
Epoch 27/100 <br>
50/50 [==============================] - 1389s 28s/step - loss: 0.9984 - categorical_accuracy: 0.5480 <br>
Epoch 28/100 <br>
50/50 [==============================] - 1388s 28s/step - loss: 0.9607 - categorical_accuracy: 0.5640 <br>
Epoch 29/100 <br>
50/50 [==============================] - 1379s 28s/step - loss: 0.9398 - categorical_accuracy: 0.5770 <br>
Epoch 30/100 <br>
50/50 [==============================] - 1368s 27s/step - loss: 0.9290 - categorical_accuracy: 0.5790 <br>
Epoch 31/100 <br>
50/50 [==============================] - 1378s 27s/step - loss: 0.9840 - categorical_accuracy: 0.5680 <br>
Epoch 32/100 <br>
50/50 [==============================] - 1381s 27s/step - loss: 0.9820 - categorical_accuracy: 0.5990 <br>
Epoch 33/100 <br>
50/50 [==============================] - 1391s 27s/step - loss: 0.9820 - categorical_accuracy: 0.6170 <br>
Epoch 34/100 <br>
50/50 [==============================] - 1402s 27s/step - loss: 0.9790 - categorical_accuracy: 0.5910 <br>
Epoch 35/100 <br>
50/50 [==============================] - 1410s 27s/step - loss: 0.9760 - categorical_accuracy: 0.5930 <br>
Epoch 36/100 <br>
50/50 [==============================] - 1420s 27s/step - loss: 0.9840 - categorical_accuracy: 0.6370 <br>
Epoch 37/100 <br>
50/50 [==============================] - 1425s 27s/step - loss: 0.9670 - categorical_accuracy: 0.6260 <br>
Epoch 38/100 <br>
50/50 [==============================] - 1431s 27s/step - loss: 0.9670 - categorical_accuracy: 0.6250 <br>
Epoch 39/100 <br>
50/50 [==============================] - 1437s 27s/step - loss: 0.9630 - categorical_accuracy: 0.6410 <br>
Epoch 40/100 <br>
50/50 [==============================] - 1444s 27s/step - loss: 0.9620 - categorical_accuracy: 0.6250 <br>
Epoch 41/100 <br>
50/50 [==============================] - 1454s 27s/step - loss: 0.9540 - categorical_accuracy: 0.6100 <br>
Epoch 42/100 <br>
50/50 [==============================] - 1459s 27s/step - loss: 0.9550 - categorical_accuracy: 0.6590 <br>
Epoch 43/100 <br>
50/50 [==============================] - 1462s 27s/step - loss: 0.9460 - categorical_accuracy: 0.6730 <br>
Epoch 44/100 <br>
50/50 [==============================] - 1465s 27s/step - loss: 0.9500 - categorical_accuracy: 0.6450 <br>
Epoch 45/100 <br>
50/50 [==============================] - 1477s 27s/step - loss: 0.9510 - categorical_accuracy: 0.6920 <br>
Epoch 46/100 <br>
50/50 [==============================] - 1489s 27s/step - loss: 0.9450 - categorical_accuracy: 0.6510 <br>
Epoch 47/100 <br>
50/50 [==============================] - 1495s 27s/step - loss: 0.9370 - categorical_accuracy: 0.6780 <br>
Epoch 48/100 <br>
50/50 [==============================] - 1505s 27s/step - loss: 0.9260 - categorical_accuracy: 0.6630 <br>
Epoch 49/100 <br>
50/50 [==============================] - 1512s 27s/step - loss: 0.9430 - categorical_accuracy: 0.6940 <br>
Epoch 50/100 <br>
50/50 [==============================] - 1523s 27s/step - loss: 0.9450 - categorical_accuracy: 0.6940 <br>
Epoch 51/100 <br>
50/50 [==============================] - 1531s 27s/step - loss: 0.9170 - categorical_accuracy: 0.6970 <br>
Epoch 52/100 <br>
50/50 [==============================] - 1546s 27s/step - loss: 0.9250 - categorical_accuracy: 0.7030 <br>
Epoch 53/100 <br>
50/50 [==============================] - 1557s 27s/step - loss: 0.9190 - categorical_accuracy: 0.7020 <br>
Epoch 54/100 <br>
50/50 [==============================] - 1562s 27s/step - loss: 0.9120 - categorical_accuracy: 0.6730 <br>
Epoch 55/100 <br>
50/50 [==============================] - 1578s 27s/step - loss: 0.9050 - categorical_accuracy: 0.6720 <br>
Epoch 56/100 <br>
50/50 [==============================] - 1582s 27s/step - loss: 0.9030 - categorical_accuracy: 0.6620 <br>
Epoch 57/100 <br>
50/50 [==============================] - 1592s 27s/step - loss: 0.9070 - categorical_accuracy: 0.6920 <br>
Epoch 58/100 <br>
50/50 [==============================] - 1591s 27s/step - loss: 0.9070 - categorical_accuracy: 0.6810 <br>
Epoch 59/10 <br>
50/50 [==============================] - 1601s 27s/step - loss: 0.9040 - categorical_accuracy: 0.7120 <br>
Epoch 60/100 <br>
50/50 [==============================] - 1612s 27s/step - loss: 0.9040 - categorical_accuracy: 0.7310 <br>
Epoch 61/100 <br>
50/50 [==============================] - 1627s 27s/step - loss: 0.8990 - categorical_accuracy: 0.7290 <br>
Epoch 62/100 <br>
50/50 [==============================] - 1631s 27s/step - loss: 0.8970 - categorical_accuracy: 0.7260 <br>
Epoch 63/100 <br>
50/50 [==============================] - 1648s 27s/step - loss: 0.8920 - categorical_accuracy: 0.7130 <br>
Epoch 64/100 <br>
50/50 [==============================] - 1659s 27s/step - loss: 0.8940 - categorical_accuracy: 0.7340 <br>
Epoch 65/100 <br>
50/50 [==============================] - 1661s 27s/step - loss: 0.8870 - categorical_accuracy: 0.7320 <br>
Epoch 66/100 <br>
50/50 [==============================] - 1672s 27s/step - loss: 0.8890 - categorical_accuracy: 0.7230 <br>
Epoch 67/100 <br>
50/50 [==============================] - 1686s 27s/step - loss: 0.8880 - categorical_accuracy: 0.7310 <br>
Epoch 68/100 <br>
50/50 [==============================] - 1694s 27s/step - loss: 0.8820 - categorical_accuracy: 0.7490 <br>
Epoch 69/100 <br>
50/50 [==============================] - 1703s 27s/step - loss: 0.8890 - categorical_accuracy: 0.7320 <br>
Epoch 70/100 <br>
50/50 [==============================] - 1705s 27s/step - loss: 0.8820 - categorical_accuracy: 0.7340 <br>
Epoch 71/100 <br>
50/50 [==============================] - 1718s 27s/step - loss: 0.8890 - categorical_accuracy: 0.7120 <br>
Epoch 72/100 <br>
50/50 [==============================] - 1713s 27s/step - loss: 0.8720 - categorical_accuracy: 0.7260 <br>
Epoch 73/100 <br>
50/50 [==============================] - 1724s 27s/step - loss: 0.8830 - categorical_accuracy: 0.7190 <br>
Epoch 74/100 <br>
50/50 [==============================] - 1735s 27s/step - loss: 0.8830 - categorical_accuracy: 0.7270 <br>
Epoch 75/100 <br>
50/50 [==============================] - 1742s 27s/step - loss: 0.8730 - categorical_accuracy: 0.7210 <br>
Epoch 76/100 <br>
50/50 [==============================] - 1756s 27s/step - loss: 0.8620 - categorical_accuracy: 0.7210 <br>
Epoch 77/100 <br>
50/50 [==============================] - 1763s 27s/step - loss: 0.8610 - categorical_accuracy: 0.7210 <br>
Epoch 78/100 <br>
50/50 [==============================] - 1774s 27s/step - loss: 0.8690 - categorical_accuracy: 0.7130 <br>
Epoch 79/100 <br>
50/50 [==============================] - 1786s 27s/step - loss: 0.8620 - categorical_accuracy: 0.7350 <br>
Epoch 80/100 <br>
50/50 [==============================] - 1790s 27s/step - loss: 0.8650 - categorical_accuracy: 0.7450 <br>
Epoch 81/100 <br>
50/50 [==============================] - 1804s 27s/step - loss: 0.8520 - categorical_accuracy: 0.7340 <br>
Epoch 82/100 <br>
50/50 [==============================] - 1813s 27s/step - loss: 0.8510 - categorical_accuracy: 0.7450 <br>
Epoch 83/100 <br>
50/50 [==============================] - 1826s 27s/step - loss: 0.8510 - categorical_accuracy: 0.7210 <br>
Epoch 84/100 <br>
50/50 [==============================] - 1838s 27s/step - loss: 0.8570 - categorical_accuracy: 0.6990 <br>
Epoch 85/100 <br>
50/50 [==============================] - 1837s 27s/step - loss: 0.8560 - categorical_accuracy: 0.7010 <br>
Epoch 86/100 <br>
50/50 [==============================] - 1845s 27s/step - loss: 0.8580 - categorical_accuracy: 0.7210 <br>
Epoch 87/100 <br>
50/50 [==============================] - 1852s 27s/step - loss: 0.8540 - categorical_accuracy: 0.7210 <br>
Epoch 88/100 <br>
50/50 [==============================] - 1862s 27s/step - loss: 0.8590 - categorical_accuracy: 0.7230 <br>
Epoch 89/100 <br>
50/50 [==============================] - 1873s 27s/step - loss: 0.8430 - categorical_accuracy: 0.7120 <br>
Epoch 90/100 <br>
50/50 [==============================] - 1889s 27s/step - loss: 0.8440 - categorical_accuracy: 0.7570 <br>
Epoch 91/100 <br>
50/50 [==============================] - 1895s 27s/step - loss: 0.8430 - categorical_accuracy: 0.7620 <br>
Epoch 92/100 <br>
50/50 [==============================] - 1910s 27s/step - loss: 0.8420 - categorical_accuracy: 0.7240 <br>
Epoch 93/100 <br>
50/50 [==============================] - 1921s 27s/step - loss: 0.8460 - categorical_accuracy: 0.7120 <br>
Epoch 94/100 <br>
50/50 [==============================] - 1936s 27s/step - loss: 0.8340 - categorical_accuracy: 0.7530 <br>
Epoch 95/100 <br>
50/50 [==============================] - 1947s 27s/step - loss: 0.8490 - categorical_accuracy: 0.7190 <br>
Epoch 96/100 <br>
50/50 [==============================] - 1951s 27s/step - loss: 0.8380 - categorical_accuracy: 0.7320 <br>
Epoch 97/100 <br>
50/50 [==============================] - 1968s 27s/step - loss: 0.8450 - categorical_accuracy: 0.7130 <br>
Epoch 98/100 <br>
50/50 [==============================] - 1972s 27s/step - loss: 0.8430 - categorical_accuracy: 0.7240 <br>
Epoch 99/100 <br>
50/50 [==============================] - 1985s 27s/step - loss: 0.8420 - categorical_accuracy: 0.7210 <br>
Epoch 100/100 <br>
50/50 [==============================] - 1998s 27s/step - loss: 0.8460 - categorical_accuracy: 0.7290 <br>
 <br>
100/100 [==============================] - 0s 1ms/step - loss: 0.8760 - categorical_accuracy: 0.6960 <br>
 <br>
Loss = 0.87605534954071045 <br>
Test Accuracy = 0.6960 <br>



Note: Require to train the whole classifier at once as checkpointing of this model is not possible. This is because the weights in our custom Keras layer are changed multiple times using multiple operations in a single batch step. Hence, saving of such a weight tensor is not yet defined in Tensorflow.

## Evaluation

The GUNN-15 model in the original paper achieved 80% accuracy for 10 class classification. In the above model we have achieved 69.60% accuracy on the test set after training on training and label subset of CIFAR-10 dataset for 3 classes. The best training accuracy achieved was 76.20% hence early stopping also could have helped. 



## Results / Conclusion

Thus, we can see that convolutional neural networks when expanded depth-wise without increasing number of weights causes good classifier of images and also does not increase the computational cost. Also, the residual network properties are seen as for such a deep network the gradients have not vanished or exploded.

## Reference

[1] GUNN is used to replace convolutional layers which is stated in paper:  Qiao, Siyuan et al. “Gradually Updated Neural Networks for Large-Scale Image Recognition.” ICML (2017).

[2]  The principle of Residual Networks also applies. He, K., Zhang, X., Ren, S., and Sun, J. Deep residual
learning for image recognition. IEEE Conference on
Computer Vision and Pattern Recognition, 2016a.

[3]  The principle of Residual Networks also applies. He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings
in deep residual networks. ECCV, 2016b.

[4]  Batch Normalization will be implemented. Ioffe, S. and Szegedy, C. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
In International Conference on Machine Learning, 2015.




