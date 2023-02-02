import os
import pandas as pd
import matplotlib.pyplot as plt


all_odors_poles = \
    ['Aminé', 'Animal', 'Boisé', 'Chimique', 'Doux', 'Empyreumatique', 'Epicé', 'Fermentaire', 'Floral', 'Frais',
     'Fruité', 'Gras', 'Lactique', 'Lactone', 'Malté', 'Minéral', 'Phénolé', 'Soufré', 'Terreux', 'Végétal']

odors_poles = ['Animal', 'Boisé', 'Chimique', 'Doux', 'Fermentaire', 'Floral', 'Frais', 'Fruité', 'Gras', 'Lactique',
               'Phénolé', 'Soufré']

data_odors_path = os.path.abspath('../data/final_odors_30.csv')


def generate_frequence_hist_odors():
    df = pd.read_csv(data_odors_path)
    frequency_odors = df.iloc[:, 2:].sum(axis=0)
    frequency_odors.sort_values(ascending=True, inplace=True)
    plt.barh(frequency_odors.index, frequency_odors.values)
    for i, v in enumerate(frequency_odors):
        plt.text(v, i, str(v))
    plt.show()


def filter_poles(poles):
    res = []
    df = pd.read_csv(data_odors_path)
    for pole in poles:
        print(pole)
        print(df.columns.tolist())
        if pole.lower() in df.columns.tolist():
            res.append(pole)
    return res


def load_data():
    """
    Fonction chargeant les données (smile & odors) depuis un fichier CSV
    :return: Retourne le graphe des labels et la liste des smiles
    """

    data = pd.read_csv(data_odors_path, sep=",", encoding="utf-8")
    smiles = data["smile"].values
    odors = data[data.columns[2:]]

    return odors, smiles
