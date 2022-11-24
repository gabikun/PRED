import utils
import numpy as np
from rdkit import Chem

# TODO completer les features : https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
features = [
    "GetSymbol",
    "GetTotalDegree",
    "GetImplicitValence",
    "GetIsAromatic"
]

def reader(molSmile):
    mol = Chem.MolFromSmiles(molSmile)
    utils.exception("mol is considered as 'None'", mol is None)

    adjacencyMatrix = createAdjacencyMatrix(mol)
    print(adjacencyMatrix)
    featureMatrix = createFeatureMatrix(mol)
    print(featureMatrix)
    # TODO function to convert the values of the featureMatrix to binaries values

def createAdjacencyMatrix(mol):
    return np.array(Chem.GetAdjacencyMatrix(mol))

def createFeatureMatrix(mol):
    result = []
    for atom in mol.GetAtoms():
        atomFeatures = []
        for feature in features:
            atomFeatures.append(getattr(atom, feature)())
        result.append(atomFeatures)

    return result



