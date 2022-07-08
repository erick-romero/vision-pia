# load and evaluate a saved model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
#labels
labels = pd.read_csv("./class_dict.csv", usecols=['class']).to_numpy()
# load model
standar_model = load_model('birdNoDataAug.h5')
dataAug_model = load_model('birdRNA.h5')
eff_model = load_model('birdeffnetb5.h5')

# summarize model.
'''
standar_model.summary()
dataAug_model.summary()
eff_model.summary()
'''
number1 = np.random.randint(314)
number2 = np.random.randint(1,5)

img_width, img_height = 224, 224
#img =  keras.utils.load_img('./bird test data/'+labels[number1][0]+'/'+str(number2)+'.jpg', target_size=(img_width, img_height))
img =  keras.utils.load_img('./bird-image.jpeg', target_size=(img_width, img_height))

x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)



model1_prediction = standar_model.predict(x)
model2_prediction = dataAug_model.predict(x)
model3_prediction = eff_model.predict(x)


score1 = tf.nn.softmax(model1_prediction[0])
score2 = tf.nn.softmax(model2_prediction[0])
score3 = tf.nn.softmax(model3_prediction[0])

print(np.argmax(score3))
print(score3[np.argmax(score3)])
print("bird image is a "+labels[number1][0])
print("Standar Net predicts: {}".format(labels[np.argmax(score1)][0]))
print("Standar Net with data augmentation predicts: {}".format(labels[np.argmax(score2)][0]))
print("EfficientNetB5 predicts: {}".format(labels[np.argmax(score3)][0]))

plt.imshow(img)
plt.title("La red neuronal predice: "+labels[np.argmax(score3)][0])
plt.show()



