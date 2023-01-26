import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


# => unique_odors de csv_reader
class_names = ['trèfle', 'acétique', 'cuit', 'citronelle', 'menthol', 'banane verte', 'bacon', 'poivre noir',
               'fermentaire', 'ylang-ylang', 'lait frais', 'marin', 'fruit exotique', 'patchouli', 'benjoin',
               'térébenthine', 'poulet rôti', 'massepain', 'menthe pouliot', 'pomme cuite', 'choucroute', 'babeurre',
               'poivre', "écorce d'orange", 'graisse de poulet', 'coumarine', 'sève de pin', 'muguet', 'beurre rance',
               'carbonisé', 'céréale', 'floral', 'concombre', 'grain torréfié', 'lys', 'cresson', 'mimosa', 'yaourt',
               'amande', 'viande cuite', 'magnolia', 'origan', 'caramel', 'chrysanthème', 'fécal', 'chimique', 'sauge',
               'poire', 'orange', 'poisson', 'orange amère', 'foin', 'foie', 'sang', 'fumée', 'huître', 'haricot vert',
               'chips', 'capucine', 'pêche', 'vinaigre', 'poivron', 'airelle', 'brandy', 'bourgeon de cassis',
               'pivoine', 'cacahuète', 'noix de macadamia', 'fromage bleu', 'asperge', 'pop corn', 'sciure', 'fenouil',
               'groseille', 'genêt', 'grain', 'violette', 'costarde', 'carambole', 'bergamote', 'acétone',
               'fruit à coque', 'durian', 'balsamique', 'salsepareille', 'zesté', 'cerise', 'ozone', 'coriandre',
               'produit pétrolier', 'métal', 'cassis', 'sucre brûlé', 'bouillon de viande', 'mandarine', 'pomme verte',
               'fromage de chèvre', 'encens', 'pin', 'persil', 'brûlé', 'freesia', 'safran', 'oignon', 'santal',
               "cire d'abeille", 'urine', 'ciste', 'doux', 'fruit de mer', 'clou de girofle', 'citrouille', 'raifort',
               'boisé', 'vert', 'jasmin', 'légume', 'oignon cuit', 'whisky', 'saumon', 'boronia', 'bois', 'tequila',
               'résineux', 'coing', 'beurre', 'genévrier', 'verveine', 'ciboulette', 'animal', 'insecte écrasé',
               'melon', 'pois de senteur', 'fève tonka', 'fromage', 'cannelle', 'myrrhe', 'anisé', 'styrax',
               'poulailler', 'tabac', 'fruité', 'ananas', 'chèvrefeuille', 'myrtille', 'friture', 'cookie', 'poireau',
               'radis', 'cive', 'datte', 'livèche', 'rhum', 'lilas', 'maïs', 'caramel au beurre', 'cèdre',
               'croûte de pain', 'cardamome', 'rose', 'wasabi', 'chèvre', "fleur d'oranger", 'linge humide', 'tomate',
               'œillet', 'aldéhyde', 'figue', 'châtaigne', 'épicé', 'œuf', 'artichaut', 'racine', 'citron vert',
               'champignon', 'coquillage', 'chamallow', 'banane', 'thym', 'amande torréfiée', 'goyave', 'peinture',
               'graisse', 'immortelle', 'alcool', 'levure', 'ail', 'poussière', 'truffe', 'salade verte', 'basilic',
               'abricot', 'rance', 'noisette', 'savon', 'vanille', 'agneau', 'pomme', 'gaz', 'aubépine', 'civette',
               'terre', 'moisi', 'orchidée', "fleur d'acacia", 'barbe à papa', 'absinthe', 'chocolat', 'raisin',
               'menthe verte', 'orge torréfiée', 'jambon', 'pain', 'cumin', 'menthé', 'galanga', 'fenugrec', 'camphre',
               'cidre', 'mangue', 'grenade', 'plastique', 'réséda', 'bière', 'solvant', 'framboise', 'héliotrope',
               'dinde cuite', 'agrume', 'amande amère', 'naphtaline', 'fruit de la passion', 'menthe poivrée',
               'confiture', 'poudré', 'sapin', 'carvone', 'grillé', 'soufré', 'lavande', 'castoréum', 'houblon',
               'poulet', 'anis', 'forêt', 'pomme de terre cuite', 'suif', 'prune', 'algue', 'cyprès', 'mouton',
               'câpres', 'phénolé', 'humus', 'putride', 'mélasse', 'kiwi', 'ambre gris', 'huile', 'luzerne', 'iodé',
               'acrylate', 'herbe', 'citron', 'café', 'feuille de tomate', 'chicorée', 'poivre vert', 'thon', 'gras',
               'frais', 'cognac', 'moutarde', 'éther', 'musc', 'réglisse', 'bouillon de légumes', 'carvi', 'moufette',
               "sirop d'érable", 'bonbon', 'noix de coco', 'cerfeuil', 'bœuf cuit', 'eau de cologne', 'miel', 'vin',
               'pâtisserie', 'ambrette', 'praline', 'pistache', 'beurre de cacahuète', 'lactique', 'noix de pécan',
               'fleur de sureau', 'palourde', 'géranium', 'baie', 'chou-fleur', 'caoutchouc', 'jacinthe',
               'noisette torréfiée', 'estragon', 'papaye', 'mûre', 'tilleul', 'mousse', 'narcisse', 'litchi',
               'chocolat noir', 'essence', 'fleur de troëne', 'romarin', 'crème', 'eucalyptus', 'pastèque', 'gingembre',
               'médical', 'bouillon', 'porc cuit', 'médicinal', 'malté', 'sueur', 'chou', 'cyclamen', 'ase fétide',
               'galbanum', 'acajou', 'parmesan', 'pain de seigle', 'ester', 'pomme de terre', 'papier', 'fruit sec',
               'graisse animale', 'noix de muscade', 'noix', 'saindoux', 'fraise', 'tubéreuse', 'bois brûlé', 'alliacé',
               'angélique', 'fleur de tilleul', 'rhubarbe', 'petits pois', 'carotte', 'limbourg', 'curry', 'menthe',
               'cacao', 'thé', 'camomille', 'noix de cajou', 'cuir', "jaune d'œuf", 'buchu', 'gardénia', 'canneberge']


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

    # Convertir les prédictions en étiquettes
    predicted_labels = np.argmax(predictions, axis=1)

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