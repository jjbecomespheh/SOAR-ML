import tensorflow.compat.v2 as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt

#Load the MNIST dataset and split it into training set and testing set
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#Visualize training set data
i=0
plt.figure(figsize=(10,10))
plt.imshow(train_images[i], cmap='gray')
plt.title("Digit: " + str(train_labels[i]))
plt.show()

#Format our data into a standard NWHC (name, width, height, channel) format
classes=10
width=28
height=28

#channel 1 for grayscale images
train_images = train_images.reshape(train_images.shape[0], width, height, 1)
test_images = test_images.reshape(test_images.shape[0], width, height, 1)
inputShape = (width, height, 1)

#Normalize data
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

#Convert integer labels to one hot encoded vector
#OHE vector is the representation of categorical variable as a binary vector (convert categorical data into numerical data using binary values)
i=0
print ("Class: ", train_labels[i])

train_labels = tf.keras.utils.to_categorical(train_labels, classes)
test_labels = tf.keras.utils.to_categorical(test_labels, classes)

print ("OHE vector: ", train_labels[i])

#Build the model
model=models.Sequential([
  layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(width, height, 1)),
  layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(10, activation='softmax')
])

model.summary()

#Compile and Train
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=10)

#to plot loss/acc graphs use the below 
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

#Evaluate and Test
test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=0)
print("Loss = {}".format(test_loss))
print("Accuracy = {}".format(test_accuracy))

#Visualizing Loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')