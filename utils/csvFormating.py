import cirpy
import csv
import pandas as pd

def old_loading():
    with open('../data/formated_odors.csv', 'w', encoding='utf-8', newline='') as writefile:
        writer = csv.writer(writefile)
        with open('../data/matrice-molecule-x-odeur-goodscents-effective-fr.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            header = header[1:]
            tens_line = -1
            for idx, row in enumerate(reader):
                smile = cirpy.resolve(row[0], 'smiles')
                odors = ""
                for i, val in enumerate(row[1:]):
                    if int(val):
                        odors += header[i] + ","
                writer.writerow([smile, odors[:-1]])

                if idx // 10 > tens_line:
                    tens_line = idx // 10
                    print(f"{tens_line}0 lines readed")



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




# MIN_DESC_ODORS = 10
# with open('../data/filtered_odors.csv', 'w', encoding='utf-8', newline='') as writefile:
#     writer = csv.writer(writefile)
#     with open('../data/matrice-molecule-x-odeur-goodscents-effective-fr.csv', 'r', encoding='utf-8') as file:
#         reader = csv.reader(file)
#         header = next(reader)
#         writer.writerow(['cas', 'smile'] + header[1:])
#
#         tens_line = -1
#         descriptors_occurence = [0]*len(header[1:])
#
#         for idx, row in enumerate(reader):
#             if idx // 10 > tens_line:
#                 tens_line = idx // 10
#                 print(f"{tens_line}0 lines readed")
#
#
#             smile = cirpy.resolve(row[0], 'smiles')
#             if not smile:
#                 continue
#
#             descriptors_occurence = [x + y for (x, y) in zip(descriptors_occurence, list(map(int, row[1:])))]
#
#             writer.writerow([row[0], smile] + row[1:])
#
#         desc_dict = dict(zip(header[1:], descriptors_occurence))
#         print(f"len dict : {len(desc_dict)}")
#         print(desc_dict)
#         filtered_desc_dict = {k: v for k, v in desc_dict.items() if v >= MIN_DESC_ODORS}
#         print(f"len filtered dict : {len(filtered_desc_dict)}")
#         print(filtered_desc_dict)
#
#
#         with open('../data/desc_odors_filtered.csv', 'w', encoding='utf-8', newline='') as resultfile:
#             resultwriter = csv.writer(resultfile)
#             for k, v in filtered_desc_dict.items():
#                 resultwriter.writerow([k, v])


#
# selected_odors_data = pd.read_csv('../data/desc_odors_filtered.csv', encoding='utf-8', header=None)
# selected_headers = ['cas', 'smile'] + selected_odors_data[selected_odors_data.columns[0]].tolist()
#
# data = pd.read_csv('../data/filtered_odors.csv', encoding='utf-8')
# print(data)
# for column in data.columns:
#     if column not in selected_headers:
#         data.drop(column, inplace=True, axis=1)
#
# data.to_csv('../data/filtered_odors2.csv', index=False)


# data = pd.read_csv('../data/filtered_odors2.csv')
# table = data[data.columns[2:]].to_numpy()
# rows_to_delete = []
# for idx, row in enumerate(table):
#     if sum(row) == 0:
#         rows_to_delete.append(idx)
# data.drop(rows_to_delete, inplace=True)
# data.reset_index(inplace=True, drop=True)
#
# data.to_csv('../data/final_odors.csv', index=False)
# delete Na et F a la main dans le csv
