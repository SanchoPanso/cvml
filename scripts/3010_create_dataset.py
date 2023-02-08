import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.file_operations.zipping import unzip_to, unpack_7z
from cvml.file_operations.main_file_ops import mv_dir, rm_dir


def main():
    src_dir = '/home/student2/datasets/source/annotations TMK 30.10'
    dst_dir = '/home/student2/datasets/raw/TMK_3010'

    zip_files = glob.glob(os.path.join(src_dir, '*.zip'))
    print(dst_dir)
    
    for zip_file in zip_files:
        unzip_to(os.path.join(src_dir, zip_file), 
                 os.path.join(dst_dir, os.path.splitext(zip_file)[0]))
    
    got_dirs = os.listdir(dst_dir)
    for got_dir in got_dirs:
        if not os.path.isdir(os.path.join(dst_dir, got_dir)):
            print(os.path.join(dst_dir, got_dir))
            continue

        subdirs = os.listdir(os.path.join(dst_dir, got_dir))
        if len(subdirs) == 1:
            mv_dir(os.path.join(dst_dir, got_dir, subdirs[0]), os.path.join(dst_dir, 'tmp'))
            rm_dir(os.path.join(dst_dir, got_dir))
            os.rename(os.path.join(dst_dir, 'tmp'), os.path.join(dst_dir, got_dir)) 
    

    
if __name__ == '__main__':
    main()
    


