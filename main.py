import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
#print("Tensorflow version: ", tf.__version__)

mnist = tf.keras.datasets.mnist

# this here is loading the dataset into x_train and y_train, which has pixel values from 0 to 255
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Defining all class names, broken into 10 types
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# data preprocessing, to conform all images into a simpler format to analyse for the NN
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.show()


# Uncomment to show how the data is formatted, not necessary for implementation but nice to help understanding
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
# previously had values from 0 - 255, now reducing as some decimal value from 0 to 1 for simplicity
train_images = train_images / 255.0
test_images = test_images / 255.0

# Should probably learn what each parameter here does lol.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)]
)

# compilation of model bnased on set of parameters, optimizer, loss function, and metrics to display
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)



