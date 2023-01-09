import deepchem as dc
import tensorflow as tf

# Taille des couches de message passing
layer_sizes = [15, 20, 27, 36]

# Fonction d'activation
activation = 'selu'

# Dimension de la partie readout
readout_dim = 175

# Création du modèle GCN
model = dc.models.GraphConvModel(len(dataset.get_shape()[1]),
                                 batch_size=batch_size,
                                 learning_rate=learning_rate,
                                 layer_sizes=layer_sizes,
                                 activation=activation,
                                 readout_dim=readout_dim)

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Entraînement du modèle
model.fit(dataset)