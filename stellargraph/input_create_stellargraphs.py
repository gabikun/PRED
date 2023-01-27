import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from rdkit import Chem
from rdkit.Chem import rdmolops


def create_stellargraphs(smiles):
    """
    Fonction permettant de générer les stellargraphs qui seront utilisés en entrée du modèle
    Un stellargraph d'une molécule contient deux informations :
        Un dataframe des caractéristiques de ses atomes
        Un dataframe représentant ses atomes sous la forme souce:target
    :param smiles: Liste contenant les smiles des molécules à traiter
    :return: Liste de StellarGraph
    """

    stellargraphs_from_mols = []

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)  # On récupère l'object molécule à l'aide de RDKit

        df_features = create_features(mol)
        df_edges = create_edges(mol)

        stellargraphs_from_mols.append(StellarGraph(df_features, df_edges))

    return stellargraphs_from_mols


def create_edges(mol):
    """
    Fonction transformant la matrice d'adjacence+identité en un dataframe de la forme (source:target)
    :param mol: La molécule à représenter
    :return: Le dataframe contenant les sources:targets pour chaque atome voisin
    """

    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)  # find fc to have the value 2 for double bonds, 3 for triple bonds
    identity_matrix = np.identity(mol.GetNumAtoms())
    id_adj = np.array(adjacency_matrix) + identity_matrix
    tmp_df = pd.DataFrame(id_adj)

    edge_list = tmp_df.stack().reset_index()
    list_source = []
    list_target = []
    for row in edge_list.values:
        if row[0] >= row[1] and row[2] == 1.0:  # If there are a connexion between two nodes and take the bond only once
            list_source.append(row[0])
            list_target.append(row[1])

    return pd.DataFrame({"source": list_source, "target": list_target})


def create_features(mol):
    """
    Fonction représentant les caractéristiques des atomes d'une molécule sous la forme d'un dataframe
    :param mol: La molécule à traiter
    :return: Le dataframe contenant les caractéristiques des atomes de la molécule
    """

    symbol_dict = {'C': 0, 'O': 1, 'N': 2, 'S': 3, 'Cl': 4, 'Br': 5, 'H': 6}
    f_symbols = []  # Contient les features "Symbol"
    f_degrees = []  # Contient les features "Degree" : nombre de voisins du sommet (tout atome confondu)
    f_implicitValences = []  # Contient les features "Implicit Valence" : nombre de H absent du smile
    f_aromatic = []  # Contient les features "Aromatic" :
    f_chirality = []  # Contient les features "Asymmetric carbon"; 0:CHI_UNSPECIFIED, 1:CHI_TETRAHEDRAL_CW, 2:CHI_TETRAHEDRAL_CCW

    for atom in mol.GetAtoms():
        f_symbols.append(symbol_dict[atom.GetSymbol()])
        f_degrees.append(atom.GetDegree())
        f_implicitValences.append(atom.GetImplicitValence())
        f_aromatic.append(int(atom.GetIsAromatic()))
        f_chirality.append(int(atom.GetChiralTag()))

    return pd.DataFrame(
        {"Symbol": f_symbols, "Degree": f_degrees, "ImplicitValence": f_implicitValences,
         "Aromatic": f_aromatic, "Chirality": f_chirality})
