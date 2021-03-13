import os
import cv2
import numpy as np
import json
import pycocotools.mask as maskUtils
import torch.utils.data as torch_data
from pathlib import Path

from ..augmentor.image_augmentor import ImageAugmenter
from .kitti_seg_eval_utils import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU



class KittiSegDataset(torch_data.Dataset):
    def __init__(self, dataset_cfg, class_names=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names: TODO
            training:
            logger:
        """
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.logger = logger
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.class_names = class_names
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.data_augmentor = ImageAugmenter(
            self.dataset_cfg.DATA_AUGMENTOR, logger=self.logger
        ) if self.training else None

        self.labels = [
            # name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
            ['unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)],
            ['ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)],
            ['rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)],
            ['out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)],
            ['static', 4, 255, 'void', 0, False, True, (0, 0, 0)],
            ['dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)],
            ['ground', 6, 255, 'void', 0, False, True, (81, 0, 81)],
            ['road', 7, 0, 'flat', 1, False, False, (128, 64, 128)],
            ['sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)],
            ['parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)],
            ['rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)],
            ['building', 11, 2, 'construction', 2, False, False, (70, 70, 70)],
            ['wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)],
            ['fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)],
            ['guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)],
            ['bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)],
            ['tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)],
            ['pole', 17, 5, 'object', 3, False, False, (153, 153, 153)],
            ['polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)],
            ['traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)],
            ['traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)],
            ['vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)],
            ['terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)],
            ['sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)],
            ['person', 24, 11, 'human', 6, True, False, (220, 20, 60)],
            ['rider', 25, 12, 'human', 6, True, False, (255, 0, 0)],
            ['car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)],
            ['truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)],
            ['bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)],
            ['caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)],
            ['trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)],
            ['train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)],
            ['motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)],
            ['bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)],
            ['license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)],
        ]
        self._label2id = {}
        self._id2label = {}
        for l in self.labels:
            self._label2id[l[0]] = l[1]
            self._id2label[l[1]] = l[0]

        self.dont_care_classes = ['unlabeled', 'ego vehicle', 'license plate', 'static', 'dynamic',
                                  'out of roi',]
        self.car_classes = ['car', 'caravan']
        self.pedestrian_classes = ['person']
        self.cyclist_classes = ['rider', 'motorcycle', 'bicycle']
        self.ground_classes = ['ground', 'terrain']
        self.vegetation_classes = ['vegetation']
        self.building_classes = ['building', 'fence', 'wall']
        self.traffic_classes = ['traffic light', 'traffic sign', 'pole', 'polegroup']
        self.sky_classes = ['sky']
        self.road_classes = ['road', 'sidewalk']
        self.other_classes = ['truck', 'bus', 'trailer', 'train']
        self.background_classes = []
        for c in self._label2id.keys():
            if c in self.dont_care_classes: continue
            if 'Car' in self.class_names and c in self.car_classes: continue
            if 'Pedestrian' in self.class_names and c not in self.pedestrian_classes: continue
            if 'Cyclist' in self.class_names and c not in self.cyclist_classes: continue
            if getattr(self.dataset_cfg, 'MULTI_CLASSES', False):
                if c in self.ground_classes or \
                    c in self.building_classes or \
                    c in self.sky_classes or \
                    c in self.other_classes\
                        : continue
            self.background_classes.append(c)

        self.class2id = {'DontCare': 255}
        self.class2color = {'DontCare': (120, 120, 120),
                            'Background': (0, 0, 0),
                            'Car': (0, 0, 142),
                            'Pedestrian': (0, 60, 100),
                            'Cyclist': (255, 0, 0),
                            'Ground': (81, 0, 81),
                            'Vegetation': (107, 142, 35),
                            'Building': (70, 70, 70),
                            'Traffic': (250, 170, 30),
                            'Sky': (70, 130, 180),
                            'Road': (244, 35, 232),
                            'Other': (152, 251, 152),}

        _id = 0
        if not getattr(self.dataset_cfg, 'MULTI_CLASSES', False):
            self.class2id['Background'] = _id
            _id += 1
        if 'Car' in self.class_names:
            self.class2id['Car'] = _id
            _id += 1
        if 'Pedestrian' in self.class_names:
            self.class2id['Pedestrian'] = _id
            _id += 1
        if 'Cyclist' in self.class_names:
            self.class2id['Cyclist'] = _id
            _id += 1

        if getattr(self.dataset_cfg, 'MULTI_CLASSES', False):
            self.class2id['Ground'] = _id
            _id += 1
            self.class2id['Vegetation'] = _id
            _id += 1
            self.class2id['Building'] = _id
            _id += 1
            self.class2id['Traffic'] = _id
            _id += 1
            self.class2id['Sky'] = _id
            _id += 1
            self.class2id['Road'] = _id
            _id += 1
            self.class2id['Other'] = _id
            _id += 1

        self.id2class = {}
        for key in self.class2id:
            self.id2class[self.class2id[key]] = key

        self.id_map = {}
        self.class_map = {}
        self.color_map = {}  # for vis

        for key in self._label2id.keys():
            if key in self.dont_care_classes:
                new_key = 'DontCare'
            elif 'Car' in self.class_names and key in self.car_classes:
                new_key = 'Car'
            elif 'Pedestrian' in self.class_names and key in self.pedestrian_classes:
                new_key = 'Pedestrian'
            elif 'Cyclist' in self.class_names and key in self.cyclist_classes:
                new_key = 'Cyclist'
            else:
                if getattr(self.dataset_cfg, 'MULTI_CLASSES', False):
                    if key in self.ground_classes:
                        new_key = 'Ground'
                    elif key in self.vegetation_classes:
                        new_key = 'Vegetation'
                    elif key in self.building_classes:
                        new_key = 'Building'
                    elif key in self.traffic_classes:
                        new_key = 'Traffic'
                    elif key in self.sky_classes:
                        new_key = 'Sky'
                    elif key in self.road_classes:
                        new_key = 'Road'
                    elif key in self.other_classes:
                        new_key = 'Other'
                    else:
                        pass
                else:
                    new_key = 'Background'

            old_id = self._label2id[key]
            new_id = self.class2id[new_key]
            self.id_map[old_id] = new_id
            self.color_map[new_id] = self.class2color[new_key]

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def get_image(self, idx):
        img_file = os.path.join(self.root_split_path, 'image_2', '%s.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)

    def get_label(self, idx):
        label_file = os.path.join(self.root_split_path, 'semantic', '%s.png' % idx)
        if not getattr(self.dataset_cfg, 'NO_LABEL', False):
            assert os.path.exists(label_file)
            return cv2.imread(label_file)[..., 0].astype(np.int32)
        else:
            return None

    def convert_label(self, label):
        for old_id in self.id_map:
            if old_id == self.id_map[old_id]:
                continue
            label[label == old_id] = self.id_map[old_id]
        return label

    def prepare_data(self, data_dict):
        if self.training:
            data_dict = self.data_augmentor.forward(data_dict)

        data_dict['image'] = data_dict['image'].astype(np.float32)

        # if self.training and len(data_dict['gt_boxes']) == 0:
        #     new_index = np.random.randint(self.__len__())
        #     return self.__getitem__(new_index)

        return data_dict

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        sample_idx = self.sample_id_list[index]

        if sample_idx.endswith('_10'):
            image = self.get_image(sample_idx)
            label = self.get_label(sample_idx)

            if self.dataset_cfg.CONVERT_LABEL and label is not None:
                label = self.convert_label(label)
        else:
            image = self.get_image(sample_idx)
            label = self.get_label(sample_idx)

        input_dict = {
            'sample_id': sample_idx,
            'image': image,
        }

        if label is not None:
            input_dict['image_seg_label'] = label

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def collate_batch(self, batch):

        batch_size = batch.__len__()
        ans_dict = {}
        ans_dict['batch_size'] = batch_size

        cnt_map = {}
        for b in batch:
            h, w = b['image'].shape[0:2]
            if f'{h}x{w}' in cnt_map:
                cnt_map[f'{h}x{w}'] += 1
            else:
                cnt_map[f'{h}x{w}'] = 1

        shapes = list(cnt_map.keys())
        argmax = shapes[0]
        for i in range(1, len(shapes)):
            if cnt_map[shapes[i]] > cnt_map[argmax]:
                argmax = shapes[i]
        h, w = [int(x) for x in argmax.split('x')]

        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                interpolation = cv2.INTER_LINEAR if key == 'image' else cv2.INTER_NEAREST
                for k in range(batch_size):
                    batch[k][key] = cv2.resize(batch[k][key].astype(np.uint8), (w, h), interpolation=interpolation)
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)
            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)
        return ans_dict

    def evaluation(self, preds, targets):
        from collections import defaultdict
        results_dict = defaultdict(list)

        metrics = ['pixel_accuracy', 'mean_accuracy', 'mean_IU', 'frequency_weighted_IU']

        for metric in metrics:
            for i in range(preds.__len__()):
                if targets[i] is not None:
                    results_dict[metric].append(eval(metric)(preds[i], targets[i]))
                else:
                    results_dict[metric] = 0.
            results_dict[metric] = np.mean(results_dict[metric])

        results_str = ''
        for metric in metrics:
            results_str += '{:25s} {:.3f}\n'.format(metric, results_dict[metric])

        return results_str, results_dict



