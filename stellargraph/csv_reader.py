import pandas as pd
from PRED.utils.utils import data_odors_path

def load_data():
    """
    Fonction chargeant les données (smile & odors) depuis un fichier CSV
    :return: Retourne le graphe des labels et la liste des smiles
    """

    data = pd.read_csv(data_odors_path, sep=",", encoding="utf-8")
    smiles = data["smile"].values
    odors = data[data.columns[2:]]

    return odors, smiles
