import pandas as pd
import numpy as np


def load_data():
    """
    Fonction chargeant les données (smile & odors) depuis un fichier CSV
    :return: Retourne le graphe des labels et la liste des smiles
    """

    data = pd.read_csv("../data/cleaned_odors.csv", sep=",", encoding="utf-8")
    odors = data["odors"].values
    smiles = data["smile"].values

    graph_labels = get_one_hot_encoding(odors)

    return pd.DataFrame(graph_labels), smiles


def get_one_hot_encoding(odors):
    """
    Fonction transformant la liste entière des odeurs en graphe des labels
    :param odors: Liste entière des odeurs du fichier CSV, toutes les odeurs pour chaque molécule
    :return: Le graphe des labels, il y a autant de lignes que de molécules et autant de colonnes que d'odeurs
    """

    # Etape 1 : Séparer les labels pour chaque molécule en utilisant split()
    odors_split = [odor.split(",") for odor in odors]

    # Etape 2 : Aplatir la liste imbriquée en une seule liste
    odors_flat = [item for sublist in odors_split for item in sublist]

    # Etape 3 : Créer un ensemble d'odeurs uniques
    unique_odors = set(odors_flat)

    # Etape 4 : Créer un dictionnaire d'odeurs uniques
    odors_dict = {odor: i for i, odor in enumerate(unique_odors)}

    # Etape 5 : Encoder les odeurs à l'aide du dictionnaire
    encoded_odors = [[odors_dict[odor] for odor in odors_split[i]] for i in range(len(odors_split))]

    # Etape 6 : Créez une matrice d'odeurs codées avec un encodage à chaud
    graph_labels = np.zeros((len(encoded_odors), len(unique_odors)))
    for i, odor_list in enumerate(encoded_odors):
        for odor in odor_list:
            graph_labels[i][odor] = 1

    return graph_labels
