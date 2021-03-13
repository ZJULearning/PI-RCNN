import torch
import torch.nn as nn
from knn_cuda import KNN
from ...utils import loss_utils


def batched_index_select(inputs, index):
    """

    :param inputs: torch.Size([batch_size, num_vertices, num_dims])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """

    batch_size, num_vertices, num_dims = inputs.shape
    k = index.shape[2]
    idx = torch.arange(0, batch_size) * num_vertices
    idx = idx.view(batch_size, -1)

    inputs = inputs.view(-1, num_dims)
    # index = index.view(batch_size, -1) + idx.type(index.dtype).to(inputs.device)
    index = index.contiguous().view(batch_size, -1) + idx.type(index.dtype).to(inputs.device)
    index = index.view(-1)

    return torch.index_select(inputs, 0, index).view(batch_size, -1, num_dims).transpose(2, 1).view(batch_size, num_dims, -1, k)


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    # TODO: pairwise distance.
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


class MLP_conv1x1(nn.Module):

    def __init__(self, channels, act_func=None, conv_func=nn.Conv2d):
        super(MLP_conv1x1, self).__init__()
        mlp = []
        for i in range(len(channels) - 1):
            mlp.append(conv_func(channels[i], channels[i+1], 1))
            if act_func is not None:
                mlp.append(act_func)
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        out = self.mlp(x)
        return out


class ContinuousConvolution(nn.Module):

    def __init__(self, in_channels, coord_channels, out_channels, K, mlp=[32, 64],
                 point_pooling=False, attentive_aggregation=False):
        super(ContinuousConvolution, self).__init__()
        self.K = K
        self.C_in = in_channels
        self.C_coord = coord_channels
        self.C_out = out_channels
        self.C_hid = mlp
        self.num_output_features = self.C_out
        self.knn = KNN(k=self.K, transpose_mode=True)
        self.kernel = MLP_conv1x1(channels=[self.C_in + self.C_coord] + self.C_hid + [self.C_out])
        self.point_pooling = point_pooling
        if self.point_pooling:
            self.num_output_features += self.C_in
        self.attentive_aggregation = attentive_aggregation
        if self.attentive_aggregation:
            self.aggr_mlp = MLP_conv1x1(channels=[self.K, 1])
            self.num_output_features += self.C_out

    def forward(self, point_features, coords):
        """
        :param points_features: B x N x C
        :param coords: B x N x 3
        :return: B x N x C'
        """
        assert self.C_coord == coords.size(-1)

        _, neighbor_idx = self.knn(coords, coords)  # B, N, K

        point_features = torch.cat((point_features, coords), dim=2)
        B, N, C = point_features.size()  # B, N, C(C_in + C_coord)
        if B == 1:
            neighbor_idx = neighbor_idx.contiguous().view(-1)  # N*K
            point_features = point_features.view(-1, C)  # N, C
            point_features = torch.index_select(point_features, 0, neighbor_idx)  # N*K, C
            point_features = point_features.view(B, -1, C)  # 1, N*K, C
            point_features = point_features.transpose(2, 1).view(B, C, -1, self.K)  # 1, C, N, K
        else:
            point_features = batched_index_select(point_features, neighbor_idx)  # B, C, N, K

        point_features[:, -self.C_coord:, :, :] -= point_features[:, -self.C_coord:, :, 0:1]  # B, C, N, K

        if self.point_pooling:
            y_pool = torch.max(point_features[:, :self.C_in, :, :], dim=3)[0]  # N, C, N

        out = self.kernel(point_features)  # B, C', N, K
        if self.attentive_aggregation:
            tmp = out.permute(0, 3, 1, 2)  # B, K, C', N
            y_aggr = self.aggr_mlp(tmp).squeeze(1)  # B, 1, C', N -> B, C', N

        out = torch.sum(out, dim=3)  # B, C', N

        if self.attentive_aggregation or self.point_pooling:
            out = torch.cat((out, y_pool, y_aggr), dim=1) # B, (C' + C' + C_in), N

        out = out.permute(0, 2, 1)   # B, N, C'
        return out


class ContConvFuseModule(nn.Module):
    def __init__(self, model_cfg):
        super(ContConvFuseModule, self).__init__()
        self.model_cfg = model_cfg
        self.cont_conv_cfg = self.model_cfg.CONT_CONV
        self.image_feature_channels = getattr(self.cont_conv_cfg, 'IMAGE_FEATURES', -1)
        self.point_feature_channels = getattr(self.cont_conv_cfg, 'POINT_FEATURES', -1)
        self.point_coord_channels = getattr(self.cont_conv_cfg, 'COORD_FEATURES', 3)
        self.cont_conv = ContinuousConvolution(in_channels=self.image_feature_channels,
                                               coord_channels=self.point_coord_channels,
                                               out_channels=self.cont_conv_cfg.OUTPUT_FEATURES,
                                               K=self.cont_conv_cfg.K,
                                               mlp=self.cont_conv_cfg.MLP,
                                               point_pooling=getattr(self.model_cfg, 'POINT_POOLING', False),
                                               attentive_aggregation=getattr(self.model_cfg, 'ATT_AGGR', False))
        self.point_feature_reshape_layer = MLP_conv1x1(
            channels=[self.point_feature_channels, self.cont_conv_cfg.OUTPUT_FEATURES], conv_func=nn.Conv1d)
        self.image_feature_reshape_layer = MLP_conv1x1(
            channels=[self.cont_conv.num_output_features, self.cont_conv_cfg.OUTPUT_FEATURES], conv_func=nn.Conv1d)

        self.forward_ret_dict = None

        if self.model_cfg.FUSE_METHODS == 'add':
            self.output_features = self.cont_conv_cfg.OUTPUT_FEATURES
        elif self.model_cfg.FUSE_METHODS == 'concat':
            self.output_features = self.cont_conv_cfg.OUTPUT_FEATURES * 2
        else:
            raise NotImplementedError

        if getattr(self.model_cfg, 'PTS_SEG_LOSS', None) is not None:
            if self.model_cfg.PTS_SEG_LOSS == 'SigmoidFocalLoss':
                self.pts_image_seg_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
            elif self.model_cfg.PTS_SEG_LOSS == 'SoftmaxFocalLoss':
                self.pts_image_seg_loss_func = None
            else:
                raise NotImplementedError
        else:
            self.pts_image_seg_loss_func = None

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        point_cls_labels = self.forward_ret_dict['point_cls_labels']

        pts_image_features = self.forward_ret_dict['pts_image_features']
        pts_image_features_flatten = pts_image_features.view(-1, pts_image_features.size(-1))

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_labels.new_zeros(point_cls_labels.size(0), pts_image_features.size(-1))
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        pts_image_seg_loss = self.pts_image_seg_loss_func(pts_image_features_flatten, one_hot_targets,
                                                          weights=cls_weights)
        pts_image_seg_loss = pts_image_seg_loss.sum()

        tb_dict.update({
            'pts_image_seg_loss': pts_image_seg_loss.item(),
        })

        return pts_image_seg_loss, tb_dict

    def forward(self, data_dict):

        if 'pred_image_seg' not in data_dict:
            image_features = data_dict['presaved_seg_features']  # B, C, H, W
        else:
            image_features = data_dict['pred_image_seg']  # B, C, H, W
        point_features = data_dict['point_features']  # B*N, C
        point_coords = data_dict['point_coords']  # B*N, 4(b, x, y, z)
        lidar_to_rect = data_dict['lidar_to_rect']  # B, 3, 4

        if 'new_point_mask' in data_dict:
            new_point_mask = data_dict['new_point_mask']
            new_point_mask = new_point_mask.bool()
        else:
            new_point_mask = None

        original_seg_features = data_dict['original_seg_features'] if 'original_seg_features' in data_dict else None

        lidar_to_rect = lidar_to_rect.permute(0, 2, 1)  # B, 4, 3
        P2 = data_dict['P2']  # B, 3, 4
        P2 = P2.permute(0, 2, 1)  # B, 4, 3
        batch_size = P2.size(0)

        pts_image_features = torch.zeros((batch_size, point_coords.size(0) // batch_size, image_features.size(1)), dtype=point_features.dtype).to(point_features.device)
        _point_coords = torch.zeros((batch_size, point_coords.size(0) // batch_size, 3), dtype=point_coords.dtype).to(point_features.device)
        for i in range(batch_size):
            _lidar_to_rect = lidar_to_rect[i]  # 4, 3
            _P2 = P2[i]  # 4, 3
            batch_mask = point_coords[:, 0] == i
            point_coords_batch = point_coords[batch_mask, 1:4]  # N, 3
            if new_point_mask is not None:
                new_point_mask_batch = new_point_mask[batch_mask]
                original_point_mask_batch = torch.logical_not(new_point_mask_batch)
                padded_pts = torch.cat((point_coords_batch[original_point_mask_batch],
                                    torch.ones((point_coords_batch[original_point_mask_batch].size(0), 1), dtype=point_coords_batch.dtype).to(point_coords_batch.device)), dim=1)  # N, 4
            else:
                padded_pts = torch.cat((point_coords_batch,
                                    torch.ones((point_coords_batch.size(0), 1), dtype=point_coords_batch.dtype).to(point_coords_batch.device)), dim=1)  # N, 4

            pts_rect = torch.mm(padded_pts.squeeze(), _lidar_to_rect)  # N, 3
            padded_pts_rect = torch.cat((pts_rect,
                                    torch.ones((pts_rect.size(0), 1), dtype=pts_rect.dtype).to(pts_rect.device)), dim=1)  # N, 4
            pts_2d = torch.mm(padded_pts_rect, _P2)  # N, 3
            pts_2d[:, 0:2] = pts_2d[:, 0:2] / padded_pts_rect[:, 2].unsqueeze(-1)
            pts_2d = pts_2d[:, 0:2]  # N, 2(u, v)
            pts_2d = pts_2d.long()

            # Although we have conduct the inverse operation(see datasets/augmentor/augmentor_utils.py and
            # datasets/augmentor/database_sampler.py) of data augmentation to guarantee that
            # the augmented 3D points can be projected into the original position on the 2D image plane.
            # The positions of corresponding 2D points has differences(maybe only 1 pixels) comparing to the situation without
            # data augmentation due to the float errors.
            if torch.max(pts_2d[:, 0]) >= image_features.size(3) or \
                    torch.max(pts_2d[:, 1]) >= image_features.size(2) or \
                    torch.min(pts_2d[:, 0]) < 0 or \
                    torch.min(pts_2d[:, 1]) < 0:
                pts_2d[pts_2d[:, 1] >= image_features.size(2), 1] = image_features.size(2) - 1
                pts_2d[pts_2d[:, 0] >= image_features.size(3), 0] = image_features.size(3) - 1
                pts_2d[pts_2d[:, 1] < 0, 1] = 0
                pts_2d[pts_2d[:, 0] < 0, 0] = 0

            if new_point_mask is not None:
                pts_image_features[i, original_point_mask_batch] = image_features[i, :, pts_2d[:, 1], pts_2d[:, 0]].permute(1, 0)
                pts_image_features[i, new_point_mask_batch] = original_seg_features[batch_mask][new_point_mask_batch, :]
            else:
                pts_image_features[i, :] = image_features[i, :, pts_2d[:, 1], pts_2d[:, 0]].permute(1, 0)

            _point_coords[i] = point_coords_batch

        if getattr(self.model_cfg, 'PTS_SEG_LOSS', None) is not None:
            ret_dict = {}
            ret_dict['pts_image_features'] = pts_image_features
            ret_dict['point_cls_labels'] = data_dict['point_cls_labels']
            self.forward_ret_dict = ret_dict

        img_features = self.cont_conv(pts_image_features, _point_coords)
        img_features = self.image_feature_reshape_layer(img_features.permute(0, 2, 1)).permute(0, 2, 1)

        point_features = point_features.view(batch_size, -1, point_features.size(-1))
        point_features = self.point_feature_reshape_layer(point_features.permute(0, 2, 1)).permute(0, 2, 1)

        if self.model_cfg.FUSE_METHODS == 'add':
            fused_features = point_features + img_features
        elif self.model_cfg.FUSE_METHODS == 'concat':
            fused_features = torch.cat((point_features, img_features), dim=2)
        else:
            raise NotImplementedError

        data_dict['point_features'] = fused_features.contiguous().view(-1, fused_features.size(-1))

        return data_dict
