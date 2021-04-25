from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


def cat_dog():
    model = load_model("vgg-16_model.h5")
    test_img1 = image.load_img('dogs vs cats/test/1/dog.8954.jpg', target_size=(224, 224))
    test_img1 = image.img_to_array(test_img1)
    test_img1 = np.expand_dims(test_img1, axis=0)
    test_img2 = image.load_img('dogs vs cats/test/0/cat.8822.jpg', target_size=(224, 224))
    test_img2 = image.img_to_array(test_img2)
    test_img2 = np.expand_dims(test_img2, axis=0)

    test_images = [test_img1, test_img2]
    result = [model.predict(test_img1), model.predict(test_img2)]
    
    for i in range(2):
        plt.imshow(test_images[i][0]/255)   
        if result[i][0][0] == 0.:
            plt.xlabel('cat')

        if result[i][0][0] == 1.:
            plt.xlabel('dog')

        plt.show()

def men_women():
    model = load_model("vgg-16_model_mw.h5")
    test_img1 = image.load_img('men vs women/test/1/00001948.jpg', target_size=(224, 224))
    test_img1 = image.img_to_array(test_img1)
    test_img1 = np.expand_dims(test_img1, axis=0)
    test_img2 = image.load_img('men vs women/test/0/00001750.jpg', target_size=(224, 224))
    test_img2 = image.img_to_array(test_img2)
    test_img2 = np.expand_dims(test_img2, axis=0)

    test_images = [test_img1, test_img2]
    result = [model.predict(test_img1), model.predict(test_img2)]
    
    for i in range(2):
        plt.imshow(test_images[i][0]/255)   
        if result[i][0][0] == 0.:
            plt.xlabel('men')

        if result[i][0][0] == 1.:
            plt.xlabel('women')

        plt.show()

sel = int(input("Select:\n1-) Cat vs Dog\n2-) Men vs Women\n"))

if sel == 1: cat_dog()
    
else: men_women()

    