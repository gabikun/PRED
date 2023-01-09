import deepchem as dc
import keras
from keras.layers import MaxPooling2D, Dense

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Obtenir les données d'entrée et les étiquettes sous forme de tableaux NumPy
X_train = train_dataset.X
y_train = train_dataset.y
X_val = valid_dataset.X
y_val = valid_dataset.y
X_test = test_dataset.X
y_test = test_dataset.y

# Définissez la forme de vos données d'entrée en utilisant la forme des tableaux NumPy
input_shape = y_train.shape[1:]

# Créez le modèle en utilisant la classe Sequential de Keras
model = keras.Sequential()

# Ajoutez 4 couches de message passing de dimensions [15, 20, 27, 36]
model.add(keras.Input(shape=input_shape))
model.add(keras.layers.Dense(15, activation='selu', name='H1'))
model.add(keras.layers.Dense(20, activation='selu', name='H2'))
model.add(keras.layers.Dense(27, activation='selu', name='H3'))
model.add(keras.layers.Dense(36, activation='selu', name='H4'))

# Ajoutez une couche de max pooling
model.add(MaxPooling2D())

# Ajoutez une couche dense avec un poids de 1 pour chaque élément de l'entrée
# Cela revient à effectuer un "Global Sum Pooling"
model.add(Dense(units=1))

# Ajoutez une couche softmax de dimension 175
model.add(Dense(units=175, activation='softmax'))

# Compilez le modèle en spécifiant une fonction de perte et un optimiseur
# model.compile(optimizer='adam', loss='categorical_crossentropy')

# Entraînez le modèle en utilisant les données d'entraînement et de validation
# model.fit(X_train, y_train, batch_size=128, epochs=5, validation_data=(X_val, y_val))

# Évaluez le modèle sur les données de test
# scores = model.evaluate(X_test, y_test, batch_size=128)
# print("Scores sur les données de test :", scores)