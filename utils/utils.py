import sys

def exception(errorMessage, cond):
    if cond:
        sys.exit(errorMessage)


odors_poles = ['Aminé', 'Animal', 'Boisé', 'Chimique', 'Doux',
               'Empyreumatique', 'Epicé', 'Fermentaire', 'Floral', 'Frais',
               'Fruité', 'Gras', 'Lactique', 'Lactone', 'Malté',
               'Minéral', 'Phénolé', 'Soufré', 'Terreux', 'Végétal']
