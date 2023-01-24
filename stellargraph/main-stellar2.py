from pygments.lexers import go
from rdkit.Chem import rdmolops
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from sklearn import model_selection

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import matplotlib.pyplot as plt

from rdkit import Chem

# Charger les données de descripteurs d'odeurs pour les molécules à partir d'un fichier CSV
data = pd.read_csv("../data/final_odors.csv", sep=",", encoding="utf-8")
odors = data["odors"].values
smiles_data = data["smile"].values

# Séparer les labels pour chaque molécule en utilisant split()
odors_split = [odor.split(",") for odor in odors]

# Aplatir la liste imbriquée en une seule liste
odors_flat = [item for sublist in odors_split for item in sublist]

# Créer un ensemble d'odeurs uniques
unique_odors = set(odors_flat)

# Créer un dictionnaire d'odeurs uniques
odor_dict = {odor: i for i, odor in enumerate(unique_odors)}

# Encoder les odeurs à l'aide du dictionnaire
encoded_odors = [[odor_dict[odor] for odor in odors_split[i]] for i in range(len(odors_split))]

# Créez une matrice d'odeurs codées avec un encodage à chaud
graph_labels = np.zeros((len(encoded_odors), len(unique_odors)))
for i, odor_list in enumerate(encoded_odors):
    for odor in odor_list:
        graph_labels[i][odor] = 1

graph_labels = pd.DataFrame(graph_labels)

# Créer des matrices d'adjacence pour les molécules
stellargraphs = [] # Contient les graphes de chaque molécule
symbol_dict = {'C':0, 'O':1, 'N':2, 'S':3, 'Cl':4, 'P':5, 'I':6, 'Na':7, 'Br':8, 'H':9, 'F':10}
for smile in smiles_data:
    f_symbols = []  # Contient les features "Symbol" de la molécule
    f_degrees = [] # Contient les features "Degree" de la molécule
    f_implicitValences = [] # Contient les features "Implicit Valence" de la molécule
    f_aromatic = [] # Contient les features "Aromatic" de la molécule

    # On récupère l'object molécule à l'aide de RDKit
    molecule = Chem.MolFromSmiles(smile)
    adjacency_matrix = rdmolops.GetAdjacencyMatrix(molecule)

    id_adj = np.array(adjacency_matrix) + np.identity(molecule.GetNumAtoms())
    graph_df = pd.DataFrame(id_adj)
    edge_list = graph_df.stack().reset_index()

    list_source = []
    list_target = []
    for row in edge_list.values:
        if row[2] == 1.0:
            list_source.append(row[0])
            list_target.append(row[1])

    # Partie graphe : source / destinataire
    dataframe_edges = pd.DataFrame(
        {"source": list_source, "target": list_target}
    )

    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        degree = atom.GetDegree()
        implicit_valence = atom.GetImplicitValence()
        aromatic = atom.GetIsAromatic()

        f_symbols.append(symbol_dict[symbol])
        f_degrees.append(degree)
        f_implicitValences.append(implicit_valence)
        if aromatic is True:
            f_aromatic.append(1)
        else:
            f_aromatic.append(0)

    # Partie features
    dataframe_features = pd.DataFrame(
        {"Symbol" : f_symbols, "Degree": f_degrees, "ImplicitValence": f_implicitValences, "Aromatic": f_aromatic}
    )

    # print(dataframe_features)

    # Assemblage de la partie graphe et features pour former un objet StellarGraph
    stellargraphs.append(StellarGraph(dataframe_features, dataframe_edges))

# Utilisez PaddedGraphGenerator pour générer les données d'entraînement
generator = PaddedGraphGenerator(stellargraphs)

# Séparer les données en ensemble d'entraînement et de test
train_data, test_data = train_test_split(data, test_size=0.2)

# Create a padded graph generator with the given graphs, node features and targets
#generator = PaddedGraphGenerator(graphs,)

def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[15, 20, 27, 36],
        activations=["selu", "selu", "selu", "selu"],
        generator=generator,
        pool_all_layers=True,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=96, activation="relu")(x_out)
    predictions = Dense(units=63, activation="relu")(predictions)
    predictions = Dense(units=360, activation="sigmoid")(predictions) # TODO AJUSTER AU NOMBRE DE DESCRIPTEURS

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model

epochs = 200  # maximum number of training epochs
folds = 2  # the number of folds for k-fold cross validation
n_repeats = 4  # the number of repeats for repeated k-fold cross validation

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)

def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc

def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen

test_accs = []
molecules_predictions_all = []
odors_predictions_all = []
smiles_all = []
n_fold = []

model = create_graph_classification_model(generator)

stratified_folds = model_selection.RepeatedKFold(
    n_splits=folds, n_repeats=n_repeats
).split(graph_labels)

for i, (train_index, test_index) in enumerate(stratified_folds):
    print(f"Training and evaluating on fold {i+1} out of {folds * n_repeats}...")
    train_gen, test_gen = get_generators(
        train_index, test_index, graph_labels, batch_size=30
    )

    history, acc = train_fold(model, train_gen, test_gen, es, epochs)

    # Récupérer les scores de prédiction pour chaque molécule
    molecules_predictions = model.predict(train_gen)
    # Récupérer les scores de prédiction pour chaque odeur
    odors_predictions = model.predict(test_gen)
    # Créer les indices de l'ensemble de données utilisé
    indices = np.arange(len(molecules_predictions))

    # Ajouter les prédictions et les smile aux listes pour tous les plis
    molecules_predictions_all.append(molecules_predictions[indices, 0])
    odors_predictions_all.append(odors_predictions[indices, 0])
    smiles_all.append(smiles_data[indices])
    for j in range(len(molecules_predictions)):
        n_fold.append(i)

    test_accs.append(acc)

print(
    f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%"
)

plt.figure(figsize=(8, 6))
plt.hist(test_accs)
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.show()

# Concaténer les listes pour tous les plis
molecules_predictions_all = np.concatenate(molecules_predictions_all)
odors_predictions_all = np.concatenate(odors_predictions_all)
smiles_all = np.concatenate(smiles_all)

# créer un DataFrame à partir des tableaux de prédiction et des smile
df = pd.DataFrame({'apprentissage': molecules_predictions_all, 'test': odors_predictions_all,
                       'smiles': smiles_all, 'n_fold': n_fold})
# créer et afficher le scatter plot final
fig = px.scatter(df, x='apprentissage', y='test', hover_name='smiles', color='n_fold')
fig.show()

