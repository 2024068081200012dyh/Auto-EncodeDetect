

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
from ...kan.kan_linear import KANLinear

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        in_channels=10

        if self.use_norm:
            #in_channels:10,out_channels:64
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.kan_linear = KANLinear(32*in_channels,32*out_channels)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
            self.kan_linear = KANLinear(32*in_channels,32*out_channels)

        self.part = 50000

    def forward(self, inputs):
        
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x_linear = torch.cat(part_linear_out, dim=0)
        else:
            x_linear = self.linear(inputs)
        x_kan = inputs.clone()
        x_kan = x_kan.reshape(inputs.shape[0],32*10)
        x_kan = self.kan_linear(x_kan)
        x_kan = x_kan.reshape(inputs.shape[0],32,64)
        x = x_linear + x_kan



        torch.backends.cudnn.enabled = False
        #x:eval_shape:torch.Size([6812, 32, 64])
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        #x:eval_shape:torch.Size([6812, 32, 64])
        x = F.relu(x)
        #x_max:eval_shape:torch.Size([6812, 1, 64])
        #>进行最大池化操作，选出每个pillar中最能代表该pillar的点
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class KPPVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        print("KPPVFE初始化")

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        #voxel_features:eval_shape:torch.Size([6812, 32, 4])
        #>当前批次数据划分为了6812个柱体/网格，每个柱体最多32个点，每个点由4个特征描述：(x,y,z,r)
        #voxel_num_points:eval_shape:torch.Size([6812])
        #>每个柱体有一个网格点(类似棋盘交叉点)，6812是柱体数量也是网格点数量
        #coords:eval_shape:torch.Size([6812, 4])
        #>coords包含了所有网格点的数据，即6812个，每个网格点由4个特征描述：(batch_index,z,y,x)
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        #points_mean:eval_shape:torch.Size([6812, 1, 3])
        #>分别对每个柱体中所有点进行求和然后分别除以该柱体中的点数得到每个柱体的平均点，注意此处只取点前三维特征，忽略反射强度r
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        #f_cluster:eval_shape:torch.Size([6812, 32, 3])
        #>每个点坐标减去其所在柱体的平均点坐标得到每个点相对平均点坐标的偏移量，即xc、yc、zc(每个点新增三个特征)
        f_cluster = voxel_features[:, :, :3] - points_mean
        #f_center:eval_shape:torch.Size([6812, 32, 3])
        #>根据每个网格点坐标计算每个柱体中心点坐标，然后将每个点坐标减去其所在柱体的中心点坐标，
        #>得到每个点相对于其所在柱体中心点坐标的偏移量，即xp、yp、zp(每个点新增三个特征)
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            #features:eval_shape:[torch.Size([6812, 32, 4]),torch.Size([6812, 32, 3]),torch.Size([6812, 32, 3])]
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        #features:eval_shape:torch.Size([6812, 32, 10])
        #>按最后一维拼接后，每个点有10个特征：(x,y,z,r,xc,yc,zc,xp,yp,zp)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        #features:eval_shape:torch.Size([6812, 32, 10])
        #>features中不存在真实点数据的特征会被填充为0，但维度不变
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        #features:eval_shape:torch.Size([6812, 64])
        #>数据经过VFE模块进去特征提取后，可以理解为6812个pillar，每个pillar由64个特征表示
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
