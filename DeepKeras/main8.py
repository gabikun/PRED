import deepchem as dc
import tensorflow as tf
import numpy as np
from tensorflow.keras import activations
from rdkit import Chem

# Jeu de données TEST
smiles = ["CC(=O)C1=NCCC1", "CC(=O)C1=NC=CS1"]
labels = ["grillé", "eau", "pop-corn"]

# Chargement du dataset tox21
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv', reload=False)
train_dataset, valid_dataset, test_dataset = tox21_datasets

# ------------------- GCN MODEL ------------------- #
# Sous-classe de `GraphConvModel` surchargeant les méthodes `build_graph_conv_layers`,
# `build_graph_pooling_layer` et `build_readout_layer`
# Cela permet de personnaliser la façon dont les couches sont construites
#
# Utilisation de SELU
# Max graph pooling
# Global sum pooling
class CustomGraphConvModel(dc.models.GraphConvModel):
    def __init__(self,
                 n_tasks,
                 graph_conv_layers,
                 dense_layer_size,
                 fc_layers,
                 output_size,
                 graph_pooling_type,
                 mode,
                 number_atom_features,
                 n_classes,
                 batch_normalize,
                 uncertainty,
                 batch_size,
                 learning_rate,
                 *args, **kwargs):
        self.graph_conv_layers = graph_conv_layers
        self.graph_pooling_type = graph_pooling_type
        self.dense_layer_size = dense_layer_size
        self.fc_layers = fc_layers
        self.output_size = output_size

        super().__init__(n_tasks=n_tasks,
                         graph_conv_layers=graph_conv_layers,
                         dense_layer_size=dense_layer_size,
                         graph_pooling_type=graph_pooling_type,
                         mode=mode,
                         number_atom_features=number_atom_features,
                         n_classes=n_classes,
                         batch_normalize=batch_normalize,
                         uncertainty=uncertainty,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         *args, **kwargs)

    def build_graph_conv_layers(self):
        """Builds graph convolution layers."""
        self.graph_convs = []
        for i in range(len(self.graph_conv_layers)):
            self.graph_convs.append(dc.models.layers.GraphConv(self.graph_conv_layers[i], activation_fn=tf.nn.selu))

    def build_graph_pooling_layer(self):
        """Builds graph pooling layer."""
        self.graph_pool = dc.models.layers.GraphPool(self.graph_pooling_type, self.batch_size)

    def build_readout_layer(self):
        """Builds readout layer."""
        # Implémentation en utilisant Keras
        self.readout = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))
        self.readout_dense = tf.keras.layers.Dense(self.dense_layer_size, activation=activations.softmax)

    def build_fc_layer(self):
        """Builds fully connected layers."""
        # Implémentation en utilisant Keras
        self.fc_dropout = tf.keras.layers.Dropout(0.47)
        self.fc_batch_norm = tf.keras.layers.BatchNormalization()
        self.fcs = []
        for i in range(len(self.fc_layers)):
            self.fcs.append(tf.keras.layers.Dense(self.fc_layers[i], activation=tf.nn.relu))

    def build_output_layer(self):
        """Builds output layer."""
        self.output = tf.keras.layers.Dense(self.output_size, activation=tf.nn.sigmoid)

    def build(self):
        """Builds the model."""
        self.build_graph_conv_layers()
        self.build_graph_pooling_layer()
        self.build_readout_layer()
        self.build_fc_layer()
        self.build_output_layer()

    def summary(self):
        """Shows the model summary."""
        print('Model: CustomGraphConvModel')
        print('Graph convolution layers:', self.graph_conv_layers)
        print('Graph pooling type:', self.graph_pooling_type)
        print('Dense layer size:', self.dense_layer_size)
        print('Fully connected layers:', self.fc_layers)
        print('Ouput layer size:', self.output_size)

# 4 couches de message passing de dimension [15, 20, 27, 36] utilisant la fonction d'activation SELU
# un max graph pooling est effectué à la fin des 4 couches de message passing
# le readout a pour dimension 175 et effectue un global sum pooling
# il est par défaut suivi d'un softmax
model = CustomGraphConvModel(
    n_tasks=len(tox21_tasks),
    graph_conv_layers=[15, 20, 27, 36],
    dense_layer_size=175,
    fc_layers=[96, 63],
    output_size=138,
    graph_pooling_type='max',
    mode='classification',
    number_atom_features=19,
    n_classes=2,
    batch_normalize=False,
    uncertainty=False,
    batch_size=50,
    learning_rate=0.001
)

# Construction du modèle
model.build()
model.summary()

model.fit(train_dataset, nb_epoch=1)

# Evaluation du modèle
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

print(model.evaluate(test_dataset, [metric], transformers))
# print(model.predict(test_dataset)[0])

# TEST
featurizer = dc.feat.ConvMolFeaturizer()
smiles = ['c1c(O)cccc1O', 'c1c(F)cccc1O', 'c1c(Cl)cccc1O']
x = featurizer.featurize([Chem.MolFromSmiles(s) for s in smiles])

print(model.predict_on_batch(x))