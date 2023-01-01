import os
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cvml.detection.dataset.annotation_converter import AnnotationConverter


if __name__ == '__main__':
    dataset_dir = r'C:\Users\Alex\Downloads\number_december1'
    save_dir = r'C:\Users\Alex\Downloads\number_december1_cropped'

    dataset_name = os.path.split(dataset_dir)[-1]
    images_dir = os.path.join(dataset_dir, 'images')
    annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_default.json')
    os.makedirs(save_dir, exist_ok=True)

    converter = AnnotationConverter()
    annotation_data = converter.read_coco(annotation_path)

    print(annotation_data.bbox_map['104'])

    for img_name in annotation_data.bbox_map.keys():
        img_path = os.path.join(images_dir, img_name + '.png')
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        bboxes = annotation_data.bbox_map[img_name]
        bbox_num = 0

        for bbox in bboxes:
            #print(bbox.get_class_id())
            if bbox.get_class_id() != 4:
                continue
            x, y, w, h = map(int, bbox.get_absolute_bounding_box())
            crop = img[y: y + h, x: x + w]
            
            save_name = f"{dataset_name}_{img_name}_{bbox_num}.png"
            save_path = os.path.join(save_dir, save_name)
            cv2.imwrite(save_path, crop)

            bbox_num += 1
            print(save_name)

