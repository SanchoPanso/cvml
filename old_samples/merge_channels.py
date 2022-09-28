import os
import cv2
import numpy as np

datasets_dir = os.path.join('E:\\PythonProjects', 'AnnotationConverter', 'datasets')

defects1_dir = os.path.join(datasets_dir, 'defects1_05082022')
defects2_dir = os.path.join(datasets_dir, 'defects2_05082022')

defects1_merged_dir = os.path.join(datasets_dir, 'defects1_merged_05082022')
defects2_merged_dir = os.path.join(datasets_dir, 'defects2_merged_05082022')


def merge_images(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)

    image_files = os.listdir(src_dir)
    image_files.sort()
    if len(image_files) == 0:
        print(f'{src_dir} has no files')
        return

    ext = os.path.splitext(image_files[0])[1]
    orig_names = set([image_file.split('_')[0] for image_file in image_files])

    print(src_dir)
    for name in orig_names:
        print(name)
        file_1 = f"{name}_1{ext}"
        file_2 = f"{name}_2{ext}"
        file_3 = f"{name}_3{ext}"

        if not os.path.exists(os.path.join(src_dir, file_1)):
            print(f"{file_1} does not exist in {src_dir}")
            continue
        if not os.path.exists(os.path.join(src_dir, file_2)):
            print(f"{file_2} does not exist in {src_dir}")
            continue
        if not os.path.exists(os.path.join(src_dir, file_3)):
            print(f"{file_3} does not exist in {src_dir}")
            continue

        img1 = cv2.imread(os.path.join(src_dir, file_1))
        img2 = cv2.imread(os.path.join(src_dir, file_2))
        img3 = cv2.imread(os.path.join(src_dir, file_3))

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

        merged_img = cv2.merge((img1, img2, img3))  # np.zeros(img1.shape, dtype=np.uint8)))
        # cv2.imshow("aaa", cv2.resize(merged_img, (600, 600)))
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(dst_dir, f"{name}{ext}"), merged_img)


if __name__ == '__main__':
    merge_images(defects1_dir, defects1_merged_dir)
    # merge_images(defects2_dir, defects2_merged_dir)

