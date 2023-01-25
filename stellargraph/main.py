from stellargraph.mapper import PaddedGraphGenerator
from csv_reader import load_data
from input_create_stellargraphs import create_stellargraphs
from model import create_graph_classification_model
from model_training import training

graph_labels, smiles = load_data()

stellargraphs = create_stellargraphs(smiles)
generator = PaddedGraphGenerator(stellargraphs)

training(generator, create_graph_classification_model(generator, graph_labels.shape[1]), graph_labels)
