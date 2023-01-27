import numpy as np
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from sklearn import model_selection
from PRED.stellargraph.model import create_graph_classification_model


def training(generator, graph_labels):
    """
    Fonction s'occupant de l'apprentissage du modèle
    :param generator: Un générateur de données d'apprentissage
    :param create_model: Une fonction permettant de créer un nouveau modèle à chaque pli
    :param graph_labels: Le dataframe contenant l'ensemble des associations molécule/odeurs
    :return: Le meilleur modèle et le jeu de test
    """

    epochs = 50  # Nombre maximal d'epochs d'entraînement
    folds = 3  # Nombre de plis pour la validation croisée
    n_repeats = 3  # Nombre de répétitions pour la validation croisée
    test_accs = []
    fold_accs = [[] for _ in range(folds)]
    best_acc = 0
    best_model = None
    best_test = []
    best_model_index = -1

    es = EarlyStopping(monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True)

    repeated_folds = model_selection.RepeatedKFold(n_splits=folds, n_repeats=n_repeats).split(graph_labels, graph_labels)

    def train_fold(model, train_gen, test_gen, es, epochs):
        """
        Fonction s'occupant de l'appentissage du modèle à chaque pli
        :param model: Le modèle
        :param train_gen: Le jeu d'apprentissage
        :param test_gen: Le jeu de test
        :param es: EarlyStopping
        :param epochs: Le nombre d'epochs
        :return: L'historique de l'apprentissage et la précision du modèle
        """

        history = model.fit(
            train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
        )
        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[model.metrics_names.index("acc")]

        return history, test_acc

    def get_generators(train_index, test_index, graph_labels, batch_size):
        """
        Fonction s'occupant de la génération des jeux d'apprentissage et de test
        :param train_index: Les index du jeu d'apprentissage complet
        :param test_index: Les index du jeu de test complet
        :param graph_labels: Le dataframe contenant l'ensemble des associations molécule/odeurs
        :param batch_size: La taille du batchsize
        :return: Un jeu d'apprentissage et de test
        """

        train_gen = generator.flow(
            train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
        )
        test_gen = generator.flow(
            test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
        )

        return train_gen, test_gen

    for i, (train_index, test_index) in enumerate(repeated_folds):
        print(f"Training and evaluating on fold {i + 1} out of {folds * n_repeats}...")
        train_gen, test_gen = get_generators(
            train_index, test_index, graph_labels, batch_size=50
        )

        model = create_graph_classification_model(generator, graph_labels.shape[1])

        history, acc = train_fold(model, train_gen, test_gen, es, epochs)

        if (acc > best_acc):
            best_acc = acc
            best_model = model
            best_test = test_gen
            best_model_index = i % folds

        test_accs.append(acc)
        fold_accs[i % folds].append(acc * 100)

    print(
        f"Accuracy over all folds mean: {np.mean(test_accs) * 100:.3}% and std: {np.std(test_accs) * 100:.2}%"
    )
    print(fold_accs)
    pyplot.boxplot(fold_accs, showmeans=True)
    pyplot.show()

    print("best model is : model " + str(best_model_index + 1))
    return best_model, best_test