import torch
from .detector3d_template import Detector3DTemplate


class PIRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        if getattr(self.model_cfg.BACKBONE_IMAGE, 'FIXED', False) and self.backbone_image is not None:
            for param in self.backbone_image.parameters():
                param.requires_grad = False

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if hasattr(self.model_cfg, 'POST_PROCESSING'):
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
            else:
                return batch_dict

    def get_training_loss(self):
        disp_dict = {}
        loss = 0.
        tb_dict = {}
        if self.backbone_image is not None and not getattr(self.model_cfg.BACKBONE_IMAGE, 'FIXED', False):
            if getattr(self.model_cfg.BACKBONE_IMAGE, 'TRAIN_WITH_LOSS', True):
                loss_seg, tb_dict = self.backbone_image.get_loss(tb_dict)
                loss += loss_seg
        if hasattr(self.model_cfg, 'FUSE_MODULE'):
            if getattr(self.model_cfg.FUSE_MODULE, 'PTS_SEG_LOSS', None) is not None:
                loss_pts_seg_loss, tb_dict = self.fuse_module.get_loss(tb_dict)
                loss += loss_pts_seg_loss
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss += loss_point
        if self.roi_head is not None:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss += loss_rcnn
        return loss, tb_dict, disp_dict
