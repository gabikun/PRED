from sklearn.manifold import TSNE
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PRED.utils.utils import data_odors_path


def suit_list(lst):
    sorted_lst = sorted(set(lst))
    index_map = {num: index for index, num in enumerate(sorted_lst)}
    return [index_map[num] for num in lst]


def show_embedding_space(x_inp, x_out, all_gen, graph_labels, model):
    # PREDICTED ODOR

    # predictions = model.predict(all_gen)
    # predicted_labels = np.argmax(predictions, axis=1)
    # unique_predicted_labels = np.unique(predicted_labels)
    # print(unique_predicted_labels)
    #
    #
    # color_predicted = suit_list(predicted_labels)
    # nb_color = max(color_predicted) + 1
    # colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])[:nb_color]
    #
    # data_odors = pd.read_csv(data_odors_path)
    # class_names = data_odors.columns[2:]
    # unique_predicted_class = [class_names[i] for i in unique_predicted_labels]

    transform = TSNE

    embedding_model = Model(inputs=x_inp, outputs=x_out)

    X = embedding_model.predict(all_gen)

    trans = transform(n_components=2, metric='cosine')
    X_reduced = trans.fit_transform(X)

    # fig, ax = plt.subplots(figsize=(7, 7))
    #
    # for g in np.unique(color_predicted):
    #     ix = np.where(color_predicted == g)
    #     ax.scatter(X_reduced[:, 0][ix], X_reduced[:, 1][ix], c=colors[g], label=unique_predicted_class[g], s=15)
    #
    # ax.legend()
    # plt.show()

    # REAL ODOR
    categories_odors = []

    labeldict = {0: 'Sans pole', 1: 'animal', 2: 'boisé', 3: 'floral', 4: 'fruité', 5: 'chimique',
                 6: 'phénolé', 7: 'gras', 8: 'lactique', 9: 'fermentaire', 10: 'soufré',
                 # 11: 'frais', 12: 'doux'
                 }

    cdict = {0: 'gray', 1: 'red', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'blue',
             6: 'cyan', 7: 'purple', 8: 'pink', 9: 'brown', 10: 'olive',
             # 11: 'black', 12: 'magenta'
             }
    pole_list = list(labeldict.values())[1:]

    for idx, row in graph_labels.iterrows():
        nb_pole = 0
        finded_pole = ""
        for pole in pole_list:
            if row[pole] == 1:
                nb_pole += 1
                finded_pole = pole

        if nb_pole == 1:
            categories_odors.append(pole_list.index(finded_pole) + 1)
        else:
            categories_odors.append(0)


    fig, ax = plt.subplots(figsize=(7, 7))

    for g in np.unique(categories_odors):
        ix = np.where(categories_odors == g)
        ax.scatter(X_reduced[:, 0][ix], X_reduced[:, 1][ix], c=cdict[g], label=labeldict[g], s=15)

    ax.legend()
    plt.show()
