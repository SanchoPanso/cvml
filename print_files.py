import os

orig_dir = r'C:\Users\Alex\Downloads\Номера+дефекты\Номера+дефекты\defects2\defects2'
new_dir = r'E:\PythonProjects\AnnotationConverter\datasets\test_defects2'

orig_names = [fn.split('.')[0] for fn in os.listdir(orig_dir)]
new_names = [fn.split('_')[0] for fn in os.listdir(new_dir)]

orig_names.sort()
new_names.sort()

print(orig_names)
print(new_names)

# 329

