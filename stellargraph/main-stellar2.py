from rdkit import Chem
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
import matplotlib.pyplot as plt


import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors



# Charger les données de descripteurs d'odeurs pour les molécules à partir d'un fichier CSV
data = pd.read_csv("molecules.csv", sep=";", encoding="utf-8")
graph_labels = data["odors"].values
smiles_data = data["smile"].values

# Créer des matrices d'adjacence pour les molécules
all_atom_features = []
edges_lists = []
symbol_dict = {'C':0, 'O':1, 'N':2, 'S':3, 'Cl':4, 'P':5, 'I':6, 'Na':7}
for smiles in smiles_data:
    molecule = Chem.MolFromSmiles(smiles)
    adjacency_matrix = rdmolops.GetAdjacencyMatrix(molecule)
    id_adj = np.array(adjacency_matrix) + np.identity(molecule.GetNumAtoms())
    graph_df = pd.DataFrame(id_adj)
    edge_list = graph_df.stack().reset_index()
    print(edge_list)
    list_source = []
    list_target = []
    for row in edge_list.values:
        if row[2] == 1.0:
            list_source.append(row[0])
            list_target.append(row[1])


    dataframe_edge = pd.DataFrame(
        {"source": list_source, "target": list_target}
    )
    print(dataframe_edge)

    atom_features = []
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        degree = atom.GetDegree()
        implicit_valence = atom.GetImplicitValence()
        aromatic = atom.GetIsAromatic()
        symbol_one_hot = [0]*8
        symbol_one_hot[symbol_dict[symbol]] = 1
        degree_one_hot = [0]*5
        degree_one_hot[degree] = 1
        implicit_valence_one_hot = [0]*5
        implicit_valence_one_hot[implicit_valence] = 1
        aromatic_one_hot = [1] if aromatic else [0]
        atom_feature = symbol_one_hot+degree_one_hot+implicit_valence_one_hot+aromatic_one_hot
        atom_features.append(atom_feature)
    all_atom_features.append(atom_features)


# Utilisez la fonction StellarGraph.from_networkx pour convertir les graphes en StellarGraph
sg_graphs = [StellarGraph.from_networkx(g) for g in graphs]

# Utilisez PaddedGraphGenerator pour générer les données d'entraînement
generator = PaddedGraphGenerator(sg_graphs)

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
    predictions = Dense(units=138, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model

epochs = 200  # maximum number of training epochs
folds = 10  # the number of folds for k-fold cross validation
n_repeats = 5  # the number of repeats for repeated k-fold cross validation

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

stratified_folds = model_selection.RepeatedStratifiedKFold(
    n_splits=folds, n_repeats=n_repeats
).split(graph_labels, graph_labels)

for i, (train_index, test_index) in enumerate(stratified_folds):
    print(f"Training and evaluating on fold {i+1} out of {folds * n_repeats}...")
    train_gen, test_gen = get_generators(
        train_index, test_index, graph_labels, batch_size=30
    )

    model = create_graph_classification_model(generator)

    history, acc = train_fold(model, train_gen, test_gen, es, epochs)

    test_accs.append(acc)

print(
    f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%"
)

plt.figure(figsize=(8, 6))
plt.hist(test_accs)
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.show()
