import numpy as np
import csv

def read_csv(file):
    csv_file = open(file)
    csv_reader_lines = csv.reader(csv_file)
    date_PyList = []
    for one_line in csv_reader_lines:
        date_PyList.append(one_line)
    bank_date_ndarray = np.array(date_PyList)
    return bank_date_ndarray

data = read_csv('bank.csv')
print(data)
# y_count = 0
# n_count = 0
# for i in data:
#     # if i[-1] == 'y':
# #     #     y_count += 1
# #     # else:
# #     #     n_count += 1
#     print(i[::-1])
#
# print(y_count,n_count)