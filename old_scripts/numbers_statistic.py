import os
from extractor import Extractor

digits_dir = r'C:\Users\Alex\Downloads\numbers_classes\numbers_classes'
digit_dirs = os.listdir(digits_dir)

digits_amount = [0 for i in range(len(digit_dirs))]
for i, digit_dir in enumerate(digit_dirs):
    cnt = 0
    files = os.listdir(os.path.join(digits_dir, digit_dir))
    for file in files:
        if not file.startswith('IMG'):
            cnt += 1
    digits_amount[i] = cnt
    print(f'Количество \'{digit_dir}\': {digits_amount[i]}')

print('Количество изображений номеров в классификации:', sum(digits_amount))


orig_image_dir = r'C:\Users\Alex\Downloads\Nomera_defecti\Nomera_defecti\numbers1\numbers1'
files = os.listdir(orig_image_dir)
print(f'Количество изображений: {len(files)}')


annot_path = r'C:\Users\Alex\Downloads\номера аннотации детекция\annotations\instances_default.json'

extr = Extractor()
data = extr.extract(annot_path)

cnt = 0
for key in data['annotations'].keys():
    lines = data['annotations'][key]
    for line in lines:
        if line[0] == 3:
            cnt += 1

# print(data['classes'])
print(f'Количество объектов \'number\': {cnt}')

