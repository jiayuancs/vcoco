import os
import json

from tqdm import tqdm
from PIL import Image


def get_paths_by_partition(partition: str):
    partition_list = ['train', 'val', 'trainval', 'test']
    assert partition in partition_list
    image_dir = dict(
        train='mscoco2014/train2014',
        val='mscoco2014/train2014',
        trainval='mscoco2014/train2014',
        test='mscoco2014/val2014'
    )
    image_path = f"/workspace/dataset/vcoco/{image_dir[partition]}"
    anno_path = f"/workspace/dataset/vcoco/instances_vcoco_{partition}.json"
    return image_path, anno_path


def save_image_size(partition: str, suffix="_size"):
    image_path, anno_path = get_paths_by_partition(partition)

    with open(anno_path, 'r') as f:
        annotations = json.load(f)
        print(f"Load: {anno_path}")

    assert "size" not in annotations.keys()

    image_size = []
    filename_list = []
    for anno in tqdm(annotations["annotations"]):
        img_path = os.path.join(image_path, anno['file_name'])
        sz = Image.open(img_path).size
        image_size.append(sz)
        filename_list.append(anno.pop('file_name'))
        anno["verb"] = anno.pop("actions")
        anno["object"] = anno.pop("objects")

    # add image_size to annotations
    annotations["size"] = image_size
    annotations["filenames"] = filename_list
    annotations["verbs"] = annotations.pop("classes")
    annotations["verb_to_object"] = annotations.pop("action_to_object")
    annotations["coco_image_id"] = annotations.pop("images")

    file_name, file_extension = os.path.splitext(anno_path)
    new_anno_path = file_name + suffix + file_extension
    with open(new_anno_path, 'w') as f:
        json.dump(annotations, f)
        print(f"Saved: {new_anno_path}")


if __name__ == '__main__':
    save_image_size(partition="train")
    save_image_size(partition="val")
    save_image_size(partition="trainval")
    save_image_size(partition="test")

    print("Done")
