import deepchem as dc
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

def global_sum_pooling(x):
    return tf.reduce_sum(x, axis=1)

global_sum_pooling_layer = keras.layers.Lambda(global_sum_pooling)

tasks, datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = datasets

n_tasks = len(tasks)
train_data = train_dataset.X
train_labels = train_dataset.y
test_data = test_dataset.X
test_labels = test_dataset.y
feature_size = train_data.shape[1]

input_shape = train_data.shape[1:]

model = keras.Sequential()

model.add(keras.layers.Dense(15, activation='selu', input_shape=input_shape))
model.add(keras.layers.Dense(20, activation='selu'))
model.add(keras.layers.Dense(27, activation='selu'))
model.add(keras.layers.Dense(36, activation='selu'))

model.add(keras.layers.GlobalMaxPooling1D())

model.add(keras.layers.Dense(175, activation='softmax'))
model.add(global_sum_pooling_layer)

model.compile(optimizer='adam', loss='categorical_crossentropy')

test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)
for task in range(n_tasks):
    print('Task {0} AUC ROC: {1:.2f}'.format(task, roc_auc_score(test_labels[:, task], predictions[:, task])))