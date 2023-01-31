from sklearn.manifold import TSNE
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np


def show_embedding_space(x_inp, x_out, all_gen, graph_labels):
    categories_odors = []
    for idx, row in graph_labels.iterrows():
        if row['animal'] == 1:
            categories_odors.append(1)
        elif row['boisé'] == 1:
            categories_odors.append(2)
        elif row['chimique'] == 1:
            categories_odors.append(3)
        elif row['doux'] == 1:
            categories_odors.append(4)
        elif row['fermentaire'] == 1:
            categories_odors.append(5)
        elif row['floral'] == 1:
            categories_odors.append(6)
        elif row['frais'] == 1:
            categories_odors.append(7)
        elif row['fruité'] == 1:
            categories_odors.append(8)
        elif row['gras'] == 1:
            categories_odors.append(9)
        elif row['lactique'] == 1:
            categories_odors.append(10)
        elif row['phénolé'] == 1:
            categories_odors.append(11)
        elif row['soufré'] == 1:
            categories_odors.append(12)
        else:
            categories_odors.append(0)

    cdict = {0: 'gray', 1: 'red', 2: 'green', 3: 'blue', 4: 'yellow', 5: 'purple',
             6: 'orange', 7: 'pink', 8: 'brown', 9: 'black', 10: 'cyan', 11: 'magenta', 12: 'olive'}
    labeldict = {0: 'Poleless', 1: 'animal', 2: 'boisé', 3: 'chimique', 4: 'doux', 5: 'fermentaire',
                 6: 'floral', 7: 'frais', 8: 'fruité', 9: 'gras', 10: 'lactique', 11: 'phénolé', 12: 'soufré'}

    transform = TSNE

    embedding_model = Model(inputs=x_inp, outputs=x_out)

    X = embedding_model.predict(all_gen)

    trans = transform(n_components=2, metric='cosine')
    X_reduced = trans.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 7))

    for g in np.unique(categories_odors):
        ix = np.where(categories_odors == g)
        ax.scatter(X_reduced[:, 0][ix], X_reduced[:, 1][ix], c=cdict[g], label=labeldict[g], s=15)

    # sc = ax.scatter(
    #     X_reduced[:, 0],
    #     X_reduced[:, 1],
    #     c=categories_odors,
    #     cmap="jet",
    #     alpha=0.7,
    # )
    #
    # ax.set(
    #     aspect="equal",
    #     xlabel="$X_1$",
    #     ylabel="$X_2$",
    #     title=f"{transform.__name__} visualization of GCN embeddings",
    # )

    ax.legend()
    plt.show()
