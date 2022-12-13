import deepchem as dc

from PRED.deepchem.dataGenerator import data_generator
from PRED.deepchem.gcnModel import MyGraphConvModel

tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

n_tasks = len(tasks)
batch_size = 100

model = dc.models.KerasModel(MyGraphConvModel(n_tasks, batch_size), loss=dc.models.losses.CategoricalCrossEntropy())
model.fit_generator(data_generator(train_dataset, n_tasks, batch_size, epochs=50))

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
print('Training set score:', model.evaluate_generator(data_generator(train_dataset, n_tasks, batch_size), [metric], transformers))
print('Test set score:', model.evaluate_generator(data_generator(test_dataset, n_tasks, batch_size), [metric], transformers))


