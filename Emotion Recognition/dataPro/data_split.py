# -*- coding: utf-8 -*-
import csv
import os

database_path = r'D:\CK_data\CK'
datasets_path = r'D:\CK_data\datasets'
csv_file = os.path.join(database_path, 'CKDatabase.csv')
train_csv = os.path.join(datasets_path, 'CKtrain.csv')
val_csv = os.path.join(datasets_path, 'CKval.csv')
test_csv = os.path.join(datasets_path, 'CKtest.csv')

with open(csv_file) as f:
    csvr = csv.reader(f)
    header = next(csvr)
    rows = [row for row in csvr]

    trn = [row[:-1] for row in rows if row[-1] == 'Training']
    csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
    print(len(trn))

    val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
    print(len(val))

    tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
    print(len(tst))