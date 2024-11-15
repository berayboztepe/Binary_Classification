This is a deep learning project that I did during my Erasmus Program with the help of my professor at Politechnika Lubelska. The goal of the project is to classify dog vs. cat images with 3 different optimizers and different epoch numbers, evaluate them, and choose the effective model.
    
The tools that will be used during this project are Anaconda to create an environment, Jupyter and Spyder to code, and some libraries such as TensorFlow, Keras, Matplotlib, Seaborn, Pandas, Numpy, and Sklearn.


# Characteristics of the data

I have 800 train images (400 for cat-400 for dog) and 600 images for test and validation both (300 for cat-300 for dog). 

**Example Images**
<br>
![](img/cat.9390.jpg)
<br>
![](img/dog.9638.jpg)


# Python Code

First, activating the environment.

```python
-conda activate tf-gpu
```
Then, I set the parameters. I will build a binary classification model, and for the loss function, I will use binary_crossentropy. Batch sizes 8 for training and validation. The model that I use is VGG16. So, the input shape is (224, 224, 3). I will build a model with 100 epochs first (50 for training and 50 for tuning), and then I will try to find the most effective epoch number for each optimizer.
```python
CLASS_MODE = 'binary'
LOSS_TYPE ='binary_crossentropy'    
CLASSES_NUMBER = 0
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8

TRAINING_EPOCHS = 50
TUNNING_EPOCHS = 50
```

The first model that I will build is using the RMSprop optimization algorithm. The metrics will be accuracy for the classification model.
```python
model.compile(loss=LOSS_TYPE,
                  optimizer=optimizers.RMSprop(lr=LR),
                  metrics=['acc'])  
```
So, I run the function.

```python
-python modelBuilding.py
```

Sizes for training and validation can be seen from the figures.
<br>
![](img/RMS-50/figure1.PNG)
<br>
![](img/RMS-50/figure2.PNG)


# RMSprop Optimization Algorithm
Now, the model has been downloaded from its github page, and the model training has been started. These are the last 5 epochs for model training.


![](img/RMS-50/last%205%20epochs%20of%20training.PNG)


After the training is completed, frozen top layers have become unfrozen, and the model tuning has been started. The reason why we do that:


![](img/Convolution_base+own_classifier.jpg)


* This is the VGG16 model without top layers. We downloaded it, and then we frozen this to build our own classifier.
```python
baseModel = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)            
model = Sequential()    
model.add(baseModel)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```


![](img/VGG16_base+own_classifier.jpg)


* This is our own classifier. We've added input layers, and with flattening, it has become 512 neurons, 256 hidden layer neurons, and their activation function is Rectifier (relu) and 1 output neuron with sigmoid activation function. With this model, we did the training. After the training was completed, we unfrozen the top layers and tuned the model with them to get a more efficient model.

```python
def freezeModel(baseModel):    
    for layer in baseModel.layers:
        layer.trainable = False 
        
freezeModel(baseModel)
```
Now, it's time to tune the model. These are the last 5 epochs for model tuning.


![](img/RMS-50/last%205%20epochs%20of%20tuning.PNG)


Let's check accuracy and loss figures and see if we can use fewer epochs to get a better result or not.


![](img/RMS-50/val_loss%20acc.PNG)


![](img/RMS-50/loss%20figure%202.PNG)


First 50 epochs for training. We need to check after the green line in the middle to see if we get a more efficient model or not. As it can be seen, at the 70. epoch, the validation loss has its lowest value. So that means there is an overfitting problem after this epoch number. Let's rerun the model with 70 epochs (50 for training and 20 for tuning) and check if we get a more efficient model or not.


![](img/RMS-50/Final/last%205%20epochs%20for%20tuning%20for%20the%20last%20rms%20model.PNG)


And the loss figure:


![](img/RMS-50/Final/loss%20figure%20for%20final%20rms%20model.PNG)


It's time to compare these 2 results. Let's see if we could improve our model or not.
```python
-python modelTesting.py
```


![](img/RMS-50/conf%20matrix.PNG)


When we check the confusion matrix, we have 16 false positives for cat and 10 for dog. So, it's just 26 mistakes of 600 images.


![](img/RMS-50/acc%20and%20losstotal%20for%20precision.PNG)


Total test accuracy is almost 0.96 which is very good, and the loss is 0.24. Let's see the model with 70 epochs.


![](img/RMS-50/Final/conf%20matrix%20for%20the%20last%20rms%20model.PNG)


It seems like mistakes have increased. 


![](img/RMS-50/Final/acc%20and%20loss%20total%20for%20precision%20for%20the%20last%20rms%20model.PNG)


**But when we check total test accuracy and test loss, our accuracy has decreased but not significantly. But when we check the difference of losses, our loss has decreased from 0.24 to 0.15, which means our loss is significantly improved with decreasing by %37.5. So, the model with 70 epochs is more efficient to use.**


# ADAM Optimization Algorithm
Now, it's time to do the same things with the ADAM optimization algorithm. First, I will try with 100 epochs again (50 for training and 50 for tuning)

```python
model.compile(loss=LOSS_TYPE,
                  optimizer=optimizers.Adam(lr=LR),
                  metrics=['acc'])  
```

```python
-python modelBuilding.py
```
Here are the last 5 epochs of training.


![](img/ADAM-50/last%205%20epochs%20for%20training%20ADAM.PNG)


Now, the top layers have been unfrozen, and these are the last 5 epochs for tuning.


![](img/ADAM-50/last%205%20epochs%20for%20tuning%20adam.PNG)


It's time to check accuracy and loss figures.


![](img/ADAM-50/accuracy%20figure%20for%20adam.PNG)


![](img/ADAM-50/loss%20figure%20for%20adam.PNG%20)


Validation loss gets its lowest value at 93. epoch. So, let's rerun the model with 93 epochs (50 training-43 tuning) and see if there is any improvement or not.
```python
-python modelBuilding.py
```


![](img/ADAM-50/Final/last%205%20epochs.PNG)


And the loss figure:


![](img/ADAM-50/Final/loss%20figure.PNG)


And it's time to compare
```python
-python modelTesting.py
```
First the confusion matrix. When we observe that, there are 26 mistakes just like the model with 100 epochs and RMSprop optimizer. 


![](img/ADAM-50/conf%20matrix%20adam.PNG)


And the total test accuracy and test loss. The same accuracy with the model with 100 epochs RMSprop optimizer, but the loss is much better.


![](img/ADAM-50/acc%20and%20loss%20total%20for%20adam.PNG)


**And these are the results for the model with 93 epochs. False positives for cats have increased so much while false positives for dogs are decreasing. We have a total 35 mistakes now. And when we check for accuracy and loss, it can be seen that accuracy decreased and loss increased. That means we have a worse model than we had before. So, the model with 93 epochs is not useful. What about if we try the second lowest value, which is in the 65. epoch? Let's try.**


![](img/ADAM-50/Final/conf%20matrix.PNG)


![](img/ADAM-50/Final/total%20acc%20and%20loss.PNG)


```python
-python modelBuilding.py
```
Again, first, last 5 epochs for model tuning.


![](img/ADAM-50/Final%202/last%205%20adam%202.PNG)


The loss figure:


![](img/ADAM-50/Final%202/loss%20adam2.PNG)


Finally, let's compare the results with the other models. First, the confusion matrix. We now have 27 mistakes. 7 false positives for cats and 20 false positives for dogs.


![](img/ADAM-50/Final%202/conf%20matrix%20adam%202.PNG)


**Now, total accuracy and total loss for testing. Accuracy has decreased very low when we compare it with the first ADAM model, but loss is significantly improved with decreasing by %14.8 (0.162-0.138). So, the model with 65 epochs is more efficient to use.**

# SGD Optimization Algorithm
Lastly, I will do the same things with the SGD optimizer and select the efficient model. First, I will build a model with 100 epochs. (50-50)
```python
model.compile(loss=LOSS_TYPE,
                  optimizer=optimizers.SGD(lr=LR),
                  metrics=['acc'])  
```

```python
-python modelBuilding.py
```
Here are the last 5 epochs for training.


![](img/SGD-50/last%205%20epochs%20for%20training.PNG)


And the last 5 epochs for tuning.


![](img/SGD-50/last%205%20epochs%20tuning.PNG)


It's time to observe the loss and accuracy figures and decide if we can rebuild a model with a lower epoch number or not.


![](img/SGD-50/acc%20figure%20sgd.PNG)


![](img/SGD-50/loss%20figure%20sgd.PNG)


It seems the validation loss gets its lowest value at 85. epoch. Let's rerun it. Last 5 epochs.


![](img/SGD-50/Final/last%205%20epochs.PNG)


The loss figure:


![](img/SGD-50/Final/loss%20figure.PNG)


So let's compare.
```python
-python modelTesting.py
```

First, the confusion matrix for the first model. We have 107 mistakes. That's quite much.


![](img/SGD-50/conf%20matrix.PNG)


And accuracy and loss for testing. Accuracy is 0.82 and loss is 0.40. This model is worse than all others. Let's see if we can improve it or not.


![](img/SGD-50/accuracy%20and%20loss%20sgd.PNG)


First, confusion matrix. Now, we have 110 mistakes.


![](img/SGD-50/Final/conf%20matrix.PNG)


And when we check testing accuracy and loss, accuracy has decreased, and loss has increased. This model is even worse.


![](img/SGD-50/Final/acc%20and%20loss%20total.PNG)


**I have tried two more different epoch numbers for this optimizer, and I got the best result with 100 epochs. We can also understand that we're not going to use this optimization algorithm due to its results. In any case, other algorithms will perform better.**

# Result Analysis and Summary
We have 7 different models with different optimization algorithms and different 
epoch numbers:

1-) RMSprop Algorithm with 100 epochs approximately:

* Accuracy: 0.96
* Loss: 0.24


2-) RMSprop Algorithm with 70 epochs approximately:

* Accuracy: 0.95
* Loss: 0.15


3-) ADAM Algorithm with 100 epochs approximately:

* Accuracy: 0.96
* Loss: 0.16 


4-) ADAM Algorithm with 93 epochs approximately:

* Accuracy: 0.94
* Loss: 0.21

5-) ADAM Algorithm with 65 epochs approximately

* Accuracy: 0.96
* Loss: 0.14

6-) SGD Algorithm with 100 epochs approximately:

* Accuracy: 0.82
* Loss: 0.41

7-) SGD Algorithm with 85 epochs approximately:

* Accuracy: 0.82
* Loss: 0.46

**So, the most effective model is the one that is built with the ADAM optimization algorithm and 65 epoch numbers. We got the highest accuracy and the lowest loss in this model. Of course, the model can be improved better with more images, different epoch numbers, different optimization algorithms, different loss and activation functions etc., but this is what I can do best with my GPU.**

# Prediction

And finally, let's do a prediction.
```python
-python prediction.py
```


![](img/2.PNG)


![](img/3.PNG)


**So, they both are predicted to be true.**

## To see the .h5 models that I've built and images that I've used

* [Images](img/Used%20Images/dogs%20vs%20cats)
* [Models](https://drive.google.com/drive/folders/1abpm0u8zIyytIAYbtpKrKE7ATWsAaVCD?usp=sharing)
