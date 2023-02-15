from stellargraph.mapper import PaddedGraphGenerator
from PRED.utils.utils import load_data
from input_create_stellargraphs import create_stellargraphs
from model_training import training
from PRED.stellargraph.confusion_matrix import show_confusion_matrix
from embedding_space import show_embedding_space


graph_labels, smiles = load_data()

stellargraphs = create_stellargraphs(smiles)
generator = PaddedGraphGenerator(stellargraphs)

best_model, x_inp, x_out = training(generator, graph_labels)

print(best_model.summary())

all_nodes = graph_labels.index
all_gen = generator.flow(all_nodes, targets=graph_labels.iloc[all_nodes].values)
all_predictions = best_model.predict(all_gen)

show_confusion_matrix(model=best_model, test_gen=all_gen)

show_embedding_space(x_inp, x_out, all_gen, graph_labels, best_model)
