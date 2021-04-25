Hello, this is a deep learning project that I did during my Erasmus Program with my professor at Politechnika Lubelska. The goal of the project is to classify dog vs cat images with 3 different optimizer and different epoch numbers, evaluate them and choose the effective model.
    
The tools will be used during this project are Anaconda to create environment, jupyter and spyder to code, some libraries such as TensorFlow, Keras, Matplotlib, Seaborn, Pandas, Numpy, Sklearn.


# Characteristics of the data

I have 800 train images (400 for cat-400 for dog), and 600 images for test and validation both (300 for cat-300 for dog). 

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
Then, I set the parameters. I will build a binary classification model and for the loss function, I will use binary_crossentropy. Batch sizes 8 for train and validation. The model which I use is VGG16. So, input shape is (224, 224, 3). I will build a model with 100 epochs first (50 for training and 50 for tuning) and then I will try to find the most effective epoch number for each optimizer.
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

The first model that I will build is using RMSprop optimization algorithm. The metrics will be accuracy for classification model.
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


Now, the model has been downloaded from its github page and the model training has been started. These are the last 5 epochs for model training.


![](img/RMS-50/last%205%20epochs%20of%20training.PNG)


After the training completed, frozen top layers has become unfrozen and the model tuning has been started. The reason why we do that:


![](img/Convolution_base+own_classifier.jpg)


* This is VGG16 model without top layers. We downloaded it and then we freeze this to build our own classifier.
```python
baseModel = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)            
model = Sequential()    
model.add(baseModel)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```


![](img/VGG16_base+own_classifier.jpg)


* This is our own classifier. We've add input layers and with flatten, it has become 512 layers, 256 hidden layers and its activation function is relu and 1 output layer with sigmoid activation function. With this model, we did the training. After the training completed, we unfrozen the top layers which and tune the model with them to get more efficient model.

```python
def freezeModel(baseModel):    
    for layer in baseModel.layers:
        layer.trainable = False 
        
freezeModel(baseModel)
```
Now, it's time to tune the model. These are the last 5 epochs for model tuning.


![](img/RMS-50/last%205%20epochs%20of%20tuning.PNG)


Let's check accuracy and loss figures and see if we can use less epochs to get better result or not.


![](img/RMS-50/val_loss%20acc.PNG)


![](img/RMS-50/loss%20figure%202.PNG)

