import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pandas as pd


# => liste des descripteurs d'odeurs
data_odors = pd.read_csv("../data/final_odors.csv")
class_names = data_odors.columns[2:]


# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=class_names_subset,
#            yticklabels=class_names_subset)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

def show_confusion_matrix(model, test_gen):
    """
    Fonctions affichant la matrice de confusion de la prédiction des odeurs
    :param model: Le modèle sur lequel on éffectue la prédiction
    :param test_gen: Le jeu de test
    """

    # Prédictions de votre modèle
    predictions = model.predict(test_gen)

    # Définir un seuil pour les prédictions
    threshold = 0.35

    # Convertir les prédictions en étiquettes en utilisant le seuil
    predicted_labels = tf.where(predictions > threshold, 1, 0)
    predicted_labels = tf.argmax(predicted_labels, axis=-1)
    # predicted_labels = np.argmax(predictions, axis=1)

    # Récupérer les étiquettes réelles
    test_labels = np.argmax(test_gen.targets, axis=1)

    # Obtenir les classes uniques dans les étiquettes réelles et les étiquettes prédites
    unique_classes = np.unique(np.concatenate((test_labels, predicted_labels)))

    # Créer un sous-ensemble de class_names qui correspond aux classes utilisées dans les étiquettes
    class_names_subset = [class_names[i] for i in unique_classes]

    # Créer la matrice de confusion
    conf_matrix = confusion_matrix(test_labels, predicted_labels, labels=unique_classes)

    # Créer le heatmap
    layout = {
        "title": "Confusion Matrix",
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"}
    }

    fig = go.Figure(data=go.Heatmap(z=conf_matrix, x=class_names_subset, y=class_names_subset, hoverongaps=False), layout=layout)
    fig.show()