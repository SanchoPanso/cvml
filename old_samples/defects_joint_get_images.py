import os
import shutil

datasets_dir = os.path.join('E:\\PythonProjects', 'AnnotationConverter', 'datasets')
defects_joint_dir = os.path.join(datasets_dir, 'defects_joint_exposure')
defects1_dir = os.path.join(datasets_dir, 'defects1_05082022')
defects2_dir = os.path.join(datasets_dir, 'defects2_05082022')

image_files = dict()
image_files['test'] = os.listdir(os.path.join(defects_joint_dir, 'test', 'images'))
image_files['train'] = os.listdir(os.path.join(defects_joint_dir, 'train', 'images'))
image_files['valid'] = os.listdir(os.path.join(defects_joint_dir, 'valid', 'images'))

for split in ['test', 'train', 'valid']:
    files = image_files[split]
    for file in files:
        name = file.split('.')[0]
        orig_name, postfix = name.split('_')
        print(split, file, orig_name, postfix, orig_name + '_3' + '.png')
        if postfix == '1':
            shutil.copy(os.path.join(defects_joint_dir, split, 'images', file),
                        os.path.join(defects1_dir, orig_name + '_3' + '.png'))
        elif postfix == '2':
            pass
            # shutil.copy(os.path.join(defects_joint_dir, split, 'images', file),
            #             os.path.join(defects2_dir, orig_name + '_3' + '.png'))
        else:
            raise ValueError(file + ' ' + postfix)

