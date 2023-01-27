from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from stellargraph.layer import GCNSupervisedGraphClassification


def create_graph_classification_model(generator, n_odors):
    """
    Fonction permettant de créer le modèle GCN en s'appuyant sur le GCNSupervisedGraphClassification
    :param generator: Le générateur de données d'entrainement
    :param n_odors: Le nombre de descripteurs d'odeurs
    :return: Le modèle GCN
    """

    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[15, 20, 27, 36],
        activations=["selu", "selu", "selu", "selu"],
        generator=generator,
        pool_all_layers=True,
        dropout=False
    )

    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=96, activation="relu")(x_out)
    predictions = Dense(units=63, activation="relu")(predictions)
    predictions = Dense(units=n_odors, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model, x_inp, x_out
