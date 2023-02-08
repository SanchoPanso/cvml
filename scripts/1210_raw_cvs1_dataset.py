import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.file_operations.zipping import unzip_to, unpack_7z
from cvml.file_operations.main_file_ops import mv_dir, rm_dir


def main():
    source_dir = r'C:\Users\Alex\Downloads\annotations TMK\annotations TMK'
    raw_dir = r'C:\Users\Alex\Downloads\TMK_CVS1'

    # create_comet_1(source_dir, raw_dir)
    create_comet_2(source_dir, raw_dir)
    # create_comet_3(source_dir, raw_dir)
    # create_comet_4(source_dir, raw_dir)
    # create_comet_5(source_dir, raw_dir)

    # create_number_0(source_dir, raw_dir)
    # create_number_1(source_dir, raw_dir)
    # create_number_2(source_dir, raw_dir)
    # create_number_3(source_dir, raw_dir)


def create_comet_1(source_dir: str, raw_dir: str):
    subdir = 'comet_1'

    # images and annotations
    unzip_to(os.path.join(source_dir, 'csv1_comet_1.zip'), 
             os.path.join(raw_dir, subdir, 'csv1_comet_1'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_1', 'images'), 
           os.path.join(raw_dir, subdir, 'images'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_1', 'annotations'), 
           os.path.join(raw_dir, subdir, 'annotations'))
    os.rmdir(os.path.join(raw_dir, subdir, 'csv1_comet_1'))

    # new annotations
    unpack_7z(os.path.join(source_dir, 'csv1_comet_annotations_24_08.7z'), 
              os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08'))
    unzip_to(os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08', 'csv1_comet_1_annotations_24_08.zip'),
             os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08', 'csv1_comet_1_annotations_24_08'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08', 'csv1_comet_1_annotations_24_08', 'annotations'),
           os.path.join(raw_dir, subdir, 'annotations'))
    rm_dir(os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08'))

    # polar_edge
    unpack_7z(os.path.join(source_dir, 'csv1_comet_1_polar_edge.7z'), 
             os.path.join(raw_dir, subdir, 'csv1_comet_1_polar_edge'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_1_polar_edge', 'csv1_comet_1_polar_edge'), 
           os.path.join(raw_dir, subdir, 'polar_edge'))
    os.rmdir(os.path.join(raw_dir, subdir, 'csv1_comet_1_polar_edge'))

    # polarization
    unpack_7z(os.path.join(source_dir, 'csv1_comet_1_polarization.7z'), 
             os.path.join(raw_dir, subdir, 'csv1_comet_1_polarization'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_1_polarization', 'csv1_comet_1_polarization'), 
           os.path.join(raw_dir, subdir, 'polarization'))
    os.rmdir(os.path.join(raw_dir, subdir, 'csv1_comet_1_polarization'))

    # visualize_images
    unzip_to(os.path.join(source_dir, 'csv1_comet_1_visualize_images.zip'), 
             os.path.join(raw_dir, subdir, 'csv1_comet_1_visualize_images'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_1_visualize_images', 'csv1_comet_1_visualize_images'), 
           os.path.join(raw_dir, subdir, 'visualize_images'))
    os.rmdir(os.path.join(raw_dir, subdir, 'csv1_comet_1_visualize_images'))

    print('comet_1 is completed')


def create_comet_2(source_dir: str, raw_dir: str):
    subdir = 'comet_2'

    # images and annotations
    unzip_to(os.path.join(source_dir, 'csv1_comet_2.zip'), 
             os.path.join(raw_dir, subdir, 'csv1_comet_2'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_2', 'images'), 
           os.path.join(raw_dir, subdir, 'images'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_2', 'annotations'), 
           os.path.join(raw_dir, subdir, 'annotations'))
    os.rmdir(os.path.join(raw_dir, subdir, 'csv1_comet_2'))

    # new annotations
    unpack_7z(os.path.join(source_dir, 'csv1_comet_annotations_24_08.7z'), 
              os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08'))
    unzip_to(os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08', 'csv1_comet_2_annotations_24_08.zip'),
             os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08', 'csv1_comet_2_annotations_24_08'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08', 'csv1_comet_2_annotations_24_08', 'annotations'),
           os.path.join(raw_dir, subdir, 'annotations'))
    rm_dir(os.path.join(raw_dir, subdir, 'csv1_comet_annotations_24_08'))


    # polar_edge
    unpack_7z(os.path.join(source_dir, 'csv1_comet_2_polar_edge.7z'), 
             os.path.join(raw_dir, subdir, 'csv1_comet_2_polar_edge'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_2_polar_edge', 'csv1_comet_2_polar_edge'), 
           os.path.join(raw_dir, subdir, 'polar_edge'))
    os.rmdir(os.path.join(raw_dir, subdir, 'csv1_comet_2_polar_edge'))

    # polarization
    unpack_7z(os.path.join(source_dir, 'csv1_comet_2.1_polarization.7z'), 
              os.path.join(raw_dir, subdir, 'csv1_comet_2.1_polarization'))
    unpack_7z(os.path.join(source_dir, 'csv1_comet_2.2_polarization.7z'), 
              os.path.join(raw_dir, subdir, 'csv1_comet_2.2_polarization'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_2.1_polarization'), 
           os.path.join(raw_dir, subdir, 'polarization'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_2.2_polarization'), 
           os.path.join(raw_dir, subdir, 'polarization'))

    # visualize_images
    unpack_7z(os.path.join(source_dir, 'csv1_comet_2.1_visualize_images.7z'), 
              os.path.join(raw_dir, subdir, 'csv1_comet_2.1_visualize_images'))
    unpack_7z(os.path.join(source_dir, 'csv1_comet_2.2_visualize_images.7z'), 
              os.path.join(raw_dir, subdir, 'csv1_comet_2.2_visualize_images'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_2.1_visualize_images'), 
           os.path.join(raw_dir, subdir, 'visualize_images'))
    mv_dir(os.path.join(raw_dir, subdir, 'csv1_comet_2.2_visualize_images'), 
           os.path.join(raw_dir, subdir, 'visualize_images'))

    print('comet_2 is completed')

if __name__ == '__main__':
    main()
    


