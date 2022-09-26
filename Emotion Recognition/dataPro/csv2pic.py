# -*- coding: utf-8 -*-
import csv
import os
from PIL import Image
import numpy as np

datasets_path = r'D:\CK_data\CK'
CK_csv = os.path.join(datasets_path, 'CKDatabase.csv')
CK_set = os.path.join(datasets_path, 'CKimages')

for save_path, csv_file in [(CK_set, CK_csv)]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num = 1
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        for i, (label, pixel) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split('+')]).reshape(48, 48)
            subfolder = os.path.join(save_path, label)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            im = Image.fromarray(pixel).convert('L')
            image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
            print(image_name)
            im.save(image_name)