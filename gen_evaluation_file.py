"""
以下代码修改自：https://github.com/s-gupta/v-coco/blob/master/vsrl_eval.py

由于原始的 v-coco 数据集评测脚本运行速度很慢，故这里将评测脚本中的预处理结果保存为文件，
以简化并加速评测过程
"""

# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2017, Saurabh Gupta 
# 
# This file is part of the VCOCO dataset hooks and is available 
# under the terms of the Simplified BSD License provided in 
# LICENSE. Please retain this notice and LICENSE if you use 
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

# vsrl_data is a dictionary for each action class:
# image_id       - Nx1
# ann_id         - Nx1
# label          - Nx1
# action_name    - string
# role_name      - ['agent', 'obj', 'instr']
# role_object_id - N x K matrix, obviously [:,0] is same as ann_id

import numpy as np
from pycocotools.coco import COCO
import json
import copy
import pickle
import os

class VCOCOeval(object):

    def __init__(self, vsrl_annot_file, coco_annot_file,
                 split_file):
        """Input:
        vslr_annot_file: path to the vcoco annotations
        coco_annot_file: path to the coco annotations
        split_file: image ids for split
        """
        self.COCO = COCO(coco_annot_file)  # 所有V-COCO图片标注数据
        self.VCOCO = _load_vcoco(vsrl_annot_file)
        self.image_ids = np.loadtxt(open(split_file, 'r'))
        # simple check
        assert np.all(np.equal(np.sort(np.unique(self.VCOCO[0]['image_id'])), self.image_ids))

        self._init_coco()
        self._init_vcoco()

    def _init_vcoco(self):
        actions = [x['action_name'] for x in self.VCOCO]  # 26 个动作名称列表
        roles = [x['role_name'] for x in self.VCOCO]  # 26 个动作对应的角色
        self.actions = actions
        self.actions_to_id_map = {v: i for i, v in enumerate(self.actions)}
        self.num_actions = len(self.actions)  # 26
        self.roles = roles

    def _init_coco(self):
        category_ids = self.COCO.getCatIds()  # 80 个物体类别编号，范围是[1, 90]
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]  # 80 个物体类别文本标签
        self.category_to_id_map = dict(zip(categories, category_ids))  # 文本到类别编号
        self.classes = ['__background__'] + categories  # 添加一个背景类别
        self.num_classes = len(self.classes)  # 81
        self.json_category_id_to_contiguous_id = {  # 将COCO的 80 个类别标签从 [1,90] 映射到 [1, 80]。0 保留给背景类别
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {  # 从 [1,80] 映射回 [1,90] 范围的 coco 标签
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}

    def _get_vcocodb(self):
        vcocodb = copy.deepcopy(self.COCO.loadImgs(self.image_ids.tolist()))
        for entry in vcocodb:
            self._prep_vcocodb_entry(entry)
            self._add_gt_annotations(entry)

            # 移除不需要的字段
            entry.pop("license")
            entry.pop("file_name")
            entry.pop("coco_url")
            entry.pop("date_captured")
            entry.pop("flickr_url")
            entry.pop("is_crowd")

        return vcocodb

    def _prep_vcocodb_entry(self, entry):
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['is_crowd'] = np.empty((0), dtype=np.bool_)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['gt_actions'] = np.empty((0, self.num_actions), dtype=np.int32)
        entry['gt_role_id'] = np.empty((0, self.num_actions, 2), dtype=np.int32)

    def _add_gt_annotations(self, entry):
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []  # valid_objs[i]是第i个box
        valid_ann_ids = []  # valid_ann_ids[i]是第i个box的标注信息编号
        width = entry['width']
        height = entry['height']
        for i, obj in enumerate(objs):
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
            x1, y1, x2, y2 = clip_xyxy_to_image(
                x1, y1, x2, y2, height, width)
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_ann_ids.append(ann_ids[i])
        num_valid_objs = len(valid_objs)
        assert num_valid_objs == len(valid_ann_ids)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_actions = -np.ones((num_valid_objs, self.num_actions), dtype=entry['gt_actions'].dtype)
        gt_role_id = -np.ones((num_valid_objs, self.num_actions, 2), dtype=entry['gt_role_id'].dtype)

        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            is_crowd[ix] = obj['iscrowd']

            # gt_actions[ix]是第ix个box的动作类别，采用one-hot编码，1表示存在该动作，可能会存在多个动作
            # gt_role_id[ix]是第ix个box的作用物box在COCO数据集中的注释编号
            gt_actions[ix, :], gt_role_id[ix, :, :] = \
                self._get_vsrl_data(valid_ann_ids[ix],
                                    valid_ann_ids, valid_objs)

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
        entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)

    def _get_vsrl_data(self, ann_id, ann_ids, objs):
        """ Get VSRL data for ann_id."""
        action_id = -np.ones((self.num_actions), dtype=np.int32)
        role_id = -np.ones((self.num_actions, 2), dtype=np.int32)
        # check if ann_id in vcoco annotations
        in_vcoco = np.where(self.VCOCO[0]['ann_id'] == ann_id)[0]
        if in_vcoco.size > 0:
            action_id[:] = 0
            role_id[:] = -1
        else:
            return action_id, role_id  # 全为-1表示该边界框内没有任何交互动作
        for i, x in enumerate(self.VCOCO):
            assert x['action_name'] == self.actions[i]
            has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
            if has_label.size > 0:
                action_id[i] = 1
                assert has_label.size == 1
                rids = x['role_object_id'][has_label]
                assert rids[0, 0] == ann_id
                for j in range(1, rids.shape[1]):
                    if rids[0, j] == 0:
                        # no role
                        continue
                    aid = np.where(ann_ids == rids[0, j])[0]
                    assert aid.size == 1
                    role_id[i, j - 1] = aid[0]
        return action_id, role_id

    def save_data(self, save_path):
        vcocodb = self._get_vcocodb()

        vcoco_data = {
            "vcocodb": vcocodb,
            "num_actions": self.num_actions,
            "roles": self.roles,
            "actions": self.actions
        }
        with open(save_path, "wb") as f:
            pickle.dump(vcoco_data, f)


def _load_vcoco(vcoco_file):
    print('loading vcoco annotations...')
    with open(vcoco_file, 'r') as f:
        vsrl_data = json.load(f)
    for i in range(len(vsrl_data)):
        vsrl_data[i]['role_object_id'] = \
            np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']), -1)).T
        for j in ['ann_id', 'label', 'image_id']:
            vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1, 1))
    return vsrl_data


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


def save_data(vcoco_root, save_dir, partition):
    assert partition in ["test", "train", "trainval", "val"]

    vsrl_annot_file = os.path.join(vcoco_root, f"data/vcoco/vcoco_{partition}.json")
    coco_file = os.path.join(vcoco_root, f"data/instances_vcoco_all_2014.json")
    split_file = os.path.join(vcoco_root, f"data/splits/vcoco_{partition}.ids")

    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vcoco_eval_{partition}.pkl")
    vcocoeval.save_data(save_path=save_path)


if __name__ == '__main__':
    # https://github.com/s-gupta/v-coco 仓库路径
    # 在执行该脚本之前，需要按照该仓库的介绍生成instances_vcoco_all_2014.json文件
    VCOCO_ROOT = "/workspace/tmp/v-coco"

    # 预处理数据保存位置
    save_dir = "./evaluation/"

    # 生成测试集上的评测数据
    save_data(vcoco_root=VCOCO_ROOT, save_dir=save_dir, partition="test")
