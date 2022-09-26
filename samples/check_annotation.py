import os
from extractor import Extractor


annot_path = r'C:\Users\Alex\Downloads\csv1_comet_2_11_08\annotations\instances_default.json'

extr = Extractor()
data = extr.extract(annot_path)

cnt = 0
for key in data['annotations'].keys():
    lines = data['annotations'][key]
    for line in lines:
        if line[0] == 3:
            cnt += 1

print(cnt)

