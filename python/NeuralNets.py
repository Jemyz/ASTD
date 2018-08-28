import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

import csv
import numpy as np

with open('reversed_vocab.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    reversed_vocab = dict(reader)
print (reversed_vocab)

with open('reversed_reviews.csv', 'r') as csv_file:
    reversed_reviews = csv.reader(csv_file)
    reversed_reviews = list(reversed_reviews)
print (reversed_reviews)

with open('labels.csv', 'r') as csv_file:
    labels = csv.reader(csv_file)
    labels = list((labels))
    labels = list(np.array(labels).flatten())
    labels = list(map(int, labels))
print (labels)

labels = np.array(labels)
labels[labels == 3] = -1


test_data = reversed_reviews[:730]
train_data= reversed_reviews[730:]

test_labels = labels[:730]
train_labels = labels[730:]

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=32)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=32)

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = len(reversed_vocab.keys())

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# model = keras.Sequential()
# model.add(keras.layers.Dense(32, input_dim=(32), activation='relu'))
# model.add(keras.layers.Dense(32, activation='relu'))
# model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model.compile(loss='cosine_proximity',optimizer='adam', metrics=['accuracy'])


x_val = train_data[:315]
partial_x_train = train_data[315:]

y_val = train_labels[:315]
partial_y_train = train_labels[315:]

print (x_val[0])
print (y_val[0])
# exit()




history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=128,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
history_dict.keys()


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()