import cirpy
import csv

# with open('../data/formated_odors.csv', 'w', encoding='utf-8', newline='') as writefile:
#     writer = csv.writer(writefile)
#     with open('../data/matrice-molecule-x-odeur-goodscents-effective-fr.csv', 'r', encoding='utf-8') as file:
#         reader = csv.reader(file)
#         header = next(reader)
#         header = header[1:]
#         tens_line = -1
#         for idx, row in enumerate(reader):
#             smile = cirpy.resolve(row[0], 'smiles')
#             odors = ""
#             for i, val in enumerate(row[1:]):
#                 if int(val):
#                     odors += header[i] + ","
#             writer.writerow([smile, odors[:-1]])
#
#             if idx // 10 > tens_line:
#                 tens_line = idx // 10
#                 print(f"{tens_line}0 lines readed")



with open('../data/cleaned_odors.csv', 'w', encoding='utf-8', newline='') as writefile:
    writer = csv.writer(writefile)
    writer.writerow(['cas', 'smile', 'odors'])
    with open('../data/formated_odors.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        with open('../data/matrice-molecule-x-odeur-goodscents-effective-fr.csv', 'r', encoding='utf-8') as casfile:
            casreader = csv.reader(casfile)
            casheader = next(casreader)

            for idx, row in enumerate(reader):
                casrow = next(casreader)
                if row[0] != '' and row[1] != '':
                    writer.writerow([casrow[0]] + row)

# TODO remettre les CAS, delete la molécule au Fluor