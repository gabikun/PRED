from sklearn.manifold import TSNE
from tensorflow.keras import Model
import matplotlib.pyplot as plt


def show_embedding_space(x_inp, x_out, all_gen, graph_labels):
    transform = TSNE

    embedding_model = Model(inputs=x_inp, outputs=x_out)

    X = embedding_model.predict(all_gen)


    trans = transform(n_components=2)
    X_reduced = trans.fit_transform(X)


    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        # c=graph_labels.values.astype("category").cat.codes,
        cmap="jet",
        alpha=0.7,
    )
    ax.set(
        aspect="equal",
        xlabel="$X_1$",
        ylabel="$X_2$",
        title=f"{transform.__name__} visualization of GCN embeddings",
    )
    plt.show()
