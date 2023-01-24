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
#
#

with open('../data/final_odors.csv', 'w', encoding='utf-8', newline='') as writefile:
    writer = csv.writer(writefile)
    with open('../data/formated_odors.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        tens_line = -1
        for idx, row in enumerate(reader):
            if row[0] != '':
                if row[1] == '':
                    writer.writerow([row[0], 'sans odeur'])
                else:
                    writer.writerow(row)

            if idx // 10 > tens_line:
                tens_line = idx // 10
                print(f"{tens_line}0 lines readed")
