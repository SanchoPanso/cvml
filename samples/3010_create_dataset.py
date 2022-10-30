import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.dataset_linking.zipping import unzip_to, unpack_7z
from cvml.dataset_linking.main_file_ops import mv_dir, rm_dir


def main():
    src_dir = r''
    dst_dir = r''

    zip_files = glob.glob(os.path.join(src_dir, '*.zip'))
    
    for zip_file in zip_files:
        unzip_to(os.path.join(src_dir, zip_file), 
                 os.path.join(dst_dir, os.path.splitext(zip_file)[0]))
    # create_comet_1(source_dir, raw_dir)
    
if __name__ == '__main__':
    main()
    


