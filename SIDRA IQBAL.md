

Image Classification using Convolutional

Neural Network (CNN)

Sidra Iqbal

School of Electrical Engineering

and Computer Science, National

University of Sciences and

Technology, Islamabad

siqbal.mscs19seecs@seecs.edu.pk

***Abstract*— Artificial Intelligence has been observing**

Relu function known as the Rectified Linear Activation

function helps the models to learn faster and better. I have

used the ReLu Activation function across all the layers

except the last layer. As this is a classification task, I have

used the Softmax function on the last layer just before the

output layer. Stochastic gradient decent has been used for

the learning rate. This is an optimization algorithm which

is used for approximating the error gradient of the

system’s current state. It uses the training set examples for

updating the model.

**an immense development in filling the gap between the**

**humans’ competencies and the machines. Researchers**

**are unceasingly exploring different aspects of this field**

**to further automate the existing world. Out of many**

**Artificial Intelligence traits, one is Computer Vision.**

**Computer Vision deals with machines to perceive the**

**world as human do. It includes various tasks such as**

**Image Classification, Image Captioning, Audio and**

**Video Recognition, Image Analysis etc. Blending the**

**Machine and Deep Learning Algorithms with**

**Computer Vision, it has reached to another level of**

**perfectness particularly in Convolutional Neural**

**Networks (CNN).**

The hyper-parameters of the stochastic gradient descent

while building a convolutional neural network should be

well tuned. The learning rate is a hyper-parameter which

is responsible for adjusting that to what extent a model

should be changed or transformed in respect to estimated

error after every time the model weights are being

updated. The learning rate ranges from 0.0 to 1.0. The

learning rate for this model is set to 0.0001.Momemtum is

used with stochastic gradient descent so that the model

converges at a faster rate and it is also responsible for

assisting the gradient to accelerate in a factual way. The

momentum for this model is set to 0.5.

***Keywords— Artificial Intelligence, Computer Vision,***

***Convolutional Neural Network.***

I. INTRODUCTION

Convolutional Neural Networks (CNN) is made up of

neurons that have associated weights and biases. These

neurons operate in a similar manner as the neurons of the

human brain functions. CNN comprises of many layers of

artificial neurons and each of these layers transforms an

activation function which is then passed to the subsequent

layers.

The model is trained with validation split of 10%.

Validation split is used for hypothetical testing. It is used

for the prediction to fit model. It comprises several hyper-

parameters that should be tuned in a manner to achieve the

highest possible accuracy. It includes training the model

using different values of parameters and observing their

accuracy and then choosing those parameters values that

yield the maximum accuracy. The hyper-parameter epoch

refers to one complete cycle of the whole training data

whereas the batch size refers to the number of training

examples or samples that are consumed during each of the

iteration cycle. While building this convolutional neural

network, different values for epoch and batch sizes have

been set and their results has been analyzed which are

being discussed in the following section.

Most decisively, three layers are used while building

CNNs namely; Convolutional Layer, Pooling Layer and

the Fully-Connected Layer. These layers are so piled in a

manner that an image with raw pixel is transformed into

the class score. Within some of these layer parameters and

hyper-parameters are so tuned to improve the accuracy of

the model.

II. METHODOLOGY

A broad convolutional neural network has been designed

using the convolutional layers. The layers of the network

had been designed using Keras. For a model to learn

complex patterns, activation functions are added to the

convolutional layers, these activation functions are of

many types each having their own distinction and use.

Mostly used activation function is the ReLu function. The

As the dataset was very large, and as the RAM should not

be fully occupied, therefore each example if being fed one

by one to the neural network.





III. RESULTS

is performing and where are the possible places it is

causing errors. The confusion matrix for the first iteration

where epoch is set to 80 and the batch\_size is set to 64 is

depicted in figure 3a.

**Model Summary:** A model summary presents the useful

information about the model. The summary of the model

in presented in figure1. It enumerates about the layers

used in the neural network and the order in which they are

stacked. It demonstrates the output of each layer. To be

precise, it visualized and summarized the whole model in

tabular form.

**Figure 2a: Confusion matrix for first hyper-**

**parameters setting.**

Correspondingly, the graph of loss and accuracy is also

been produced after each cycle. This graph expresses the

errors in our data. It also measures the accuracy in terms

of the error that we are making in our data. The graph of

loss and Accuracy for the first iteration where epoch is set

to 80 and the batch\_size is set to 64 is depicted in figure

3b.

**Figure 1: Model Summary**

**Setting Parameters:** the hyper-parameters are tuned

during each of the iteration, while changing their values.

**First Iteration:** On the first iteration the setting of the

hyper-parameters to train the model is given as:

*history = model.fit(X\_train,y\_train,epochs=90,batch\_size*

*=128, verbose=1,validation\_split=0.10)*

when the model was trained using this setting of the

hyper-parameters, the accuracy achieved was 0.6690

which is **66.90%.** It is represented in figure 2a and 2b.

**Second Iteration:** On the first iteration the setting of the

hyper-parameters to train the model is given as:

*history = model.fit(X\_train,y\_train,epochs=80,batch\_size*

*=64, verbose=1,validation\_split=0.10)*

when the model was trained using this setting of the

hyper-parameters, the accuracy achieved was 0.7087

which is **70.87%.**

The confusion matrix is used to represent the summary of

your classification model about predicting results. The

confusion matrix has been generated after each time the

model is trained. It explains how our classification model





**Figure 3b: Graph of loss and accuracy for first hyper-**

**parameters setting.**

IV. DISCUSSION AND CONCLUSION

The convolutional neural network has been trained using

different hyper-parameters. The stochastic gradient

descent has been used for learning rate. A broad CNN has

been build using different convolutional layers, also max

pooling has been applied. Activation functions such as

ReLu and Softmax has been used for this classification

task. The parameters have been tuned to achieve the

maximum accuracy.

V. GITHUB REPOSITORY LINK

**Figure 3a: Confusion matrix for first hyper-**

**parameters setting.**

**Figure 3b: Graph of loss and accuracy for first hyper-**

**parameters setting.**









