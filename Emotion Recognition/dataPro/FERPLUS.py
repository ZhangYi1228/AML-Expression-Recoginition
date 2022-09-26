import csv
import numpy as np


label1 = []
with open('D:/FERPLUS/FERPlus/fer2013plus.csv', 'r') as f:
    reader = csv.reader(f)
    # print(type(reader))

    for row in reader:
        # print(row)
        if row[0] != '':
            label1.append(row)
# Create file object
f = open('D:/FERPLUS/FERPlus/fer2013plus1.csv','w',encoding='utf-8',newline = '')

# Build a csv write object based on the file object
csv_writer = csv.writer(f)

# build list header
csv_writer.writerow(["emotion","pixels","Usage"])

# write csv file content
for num in label1:
    csv_writer.writerow(num)



f.close()





