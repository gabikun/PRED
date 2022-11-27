import molReader
import gcn

adjacencyMatrix, binaryFeatureMatrix, regularizationMatrix = molReader.reader('COC1=CC=CC(=C1O)C=NC(CCSC)C(=O)O')
print(regularizationMatrix)

model = gcn.GCN(19)
print(model)
