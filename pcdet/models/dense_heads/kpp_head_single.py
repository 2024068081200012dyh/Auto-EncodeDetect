
import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from ..kan.kan_linear import KANLinear

class KPPHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        print("KPPHeadSingle初始化")

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        #spatial_features_2d:eval_shape:torch.Size([1, 384, 248, 216])
        #>Backbone特征提取和融合得到的数据，也可以看成248*216的伪图像，每个点(根据后面代码理解，此处一个点即一个候选框目标)384个特征维度
        spatial_features_2d = data_dict['spatial_features_2d']
        #cls_preds:eval_shape:torch.Size([1, 18, 248, 216])
        #>检测目标有6个先验框，每个先验框类别有三类，所以每个点有18个特征维度
        cls_preds = self.conv_cls(spatial_features_2d)
        #box_preds:eval_shape:torch.Size([1, 42, 248, 216])
        #>检测目标有6个先验框，每个先验框包含7个参数，[x,y,z,w,l,h,θ]，所以每个点有42个特征维度
        box_preds = self.conv_box(spatial_features_2d)
        #cls_preds:eval_shape:torch.Size([1, 248, 216, 18])
        #>调整了下维度顺序，现在维度含义：[批次大小,长,宽,特征]
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        #box_preds:eval_shape:torch.Size([1, 248, 216, 42])
        #>调整了下维度顺序，现在维度含义：[批次大小,长,宽,特征]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            #dir_cls_preds:eval_shape:torch.Size([1, 12, 248, 216])
            #>检测目标有6个先验框，每个先验框要预测两个方向中其中一个方向，所以每个点有12个特征维度
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            #dir_cls_preds:eval_shape:torch.Size([1, 248, 216, 12])
            #>调整了下维度顺序，现在维度含义：[批次大小,长,宽,特征]
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            #batch_cls_preds:eval_shape:torch.Size([1, 321408, 3])
            #>维度含义：[批次大小,候选框数量,检测框类别]
            #batch_box_preds:eval_shape:torch.Size([1, 321408, 7])
            #>维度含义：[批次大小,候选框数量,检测框空间信息]
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
