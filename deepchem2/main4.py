import deepchem as dc
import matplotlib.pyplot as plot
import numpy as np
from PRED.deepchem2.gcnModel import MyGraphConvModel
from PRED.deepchem2.dataGenerator import data_generator
from deepchem.models.fcnet import MultitaskClassifier

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv', reload=False)
train_dataset, valid_dataset, test_dataset = tox21_datasets

n_tasks = len(tox21_tasks)
batch_size = 100

# GCN model
model = dc.models.KerasModel(MyGraphConvModel(n_tasks, batch_size), loss=dc.models.losses.CategoricalCrossEntropy())

# Loss
num_epochs = 10
losses = []
for i in range(num_epochs):
    loss = model.fit_generator(data_generator(train_dataset, n_tasks, batch_size, epochs=1))
    print("Epoch %d loss: %f" % (i, loss))
    losses.append(loss)

# Plot the Loss
plot.ylabel("Loss")
plot.xlabel("Epoch")
x = range(num_epochs)
y = losses
plot.scatter(x, y)
plot.show()

# Performance of the model
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")

print("Evaluating model")
train_scores = model.evaluate_generator(data_generator(train_dataset, n_tasks, batch_size), [metric], transformers)
print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
valid_scores = model.evaluate_generator(data_generator(train_dataset, n_tasks, batch_size), [metric], transformers)
print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])




MTCmodel = MultitaskClassifier(n_tasks=175, n_features=138, layer_sizes=[96, 63], dropouts=0.47, activation_fns='relu', n_classes=138)
