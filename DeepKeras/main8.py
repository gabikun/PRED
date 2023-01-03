import deepchem as dc
import tensorflow as tf
import numpy as np
from tensorflow.keras import activations

# Chargement du dataset tox21
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
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
        self.readout_dense = tf.keras.layers.Dense(self.n_tasks, activation=activations.softmax)

    def build(self):
        """Builds the model."""
        self.build_graph_conv_layers()
        self.build_graph_pooling_layer()
        self.build_readout_layer()

    def summary(self):
        """Shows the model summary."""
        print('Model: CustomGraphConvModel')
        print('Graph convolution layers:', self.graph_conv_layers)
        print('Graph pooling type:', self.graph_pooling_type)
        print('Dense layer size:', self.dense_layer_size)

# 4 couches de message passing de dimension [15, 20, 27, 36] utilisant la fonction d'activation SELU
# un max graph pooling est effectué à la fin des 4 couches de message passing
# le readout a pour dimension 175 et effectue un global sum pooling
# il est par défaut suivi d'un softmax
model = CustomGraphConvModel(
    n_tasks=len(tox21_tasks),
    graph_conv_layers=[15, 20, 27, 36],
    dense_layer_size=175,
    graph_pooling_type='max',
    mode='classification',
    number_atom_features=19,
    n_classes=2,
    batch_normalize=True,
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
