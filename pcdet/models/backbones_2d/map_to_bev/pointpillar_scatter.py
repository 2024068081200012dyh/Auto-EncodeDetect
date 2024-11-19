import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        #pillar_features:eval_shape:torch.Size([6812, 64])
        #>6812个pillar，每个pillar由64个特征表示
        #coords:eval_shape:torch.Size([6812, 4])
        #>6812个网格点，每个网格点由4个特征表示
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        #batch_size:eval_shape:1
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            #>spatial_feature:eval_shape:torch.Size([64, 432*496*1=214272]),伪图像被拉成一维了
            spatial_feature = torch.zeros(
                self.num_bev_features, #64
                self.nz * self.nx * self.ny, #nx=432,ny=496,nz=1,伪图像大小为：432*496*1
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            #indices:eval_shape:torch.Size([6812])
            #>当前批次pillar的网格点有6182个，此处获取6182个pillar在拉成一维的伪图像上的索引位置
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            #pillars:eval_shape:torch.Size([6812, 64])
            #>由于批次大小为1，所以当前批次pillar即所有pillar
            pillars = pillar_features[batch_mask, :]
            #pillars:eval_shape:torch.Size([64, 6812])
            #>由于spatial_feature的特征表示在第0维，因此需要转置
            pillars = pillars.t()
            #分别将6812个pillar根据其网格点计算得到的其在一维伪图像上的索引保存在spatial_feature中
            spatial_feature[:, indices] = pillars
            #加入批次空间特征列表，但是由于batch_size=1,因此列表里就只有这一个一维伪图像
            batch_spatial_features.append(spatial_feature)
        #把当前批次得到的batch_size个数据映射成的batch_size个一维伪图像在第0个维度上堆叠起来，此处设置batch_size=1
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        #batch_spatial_features:eval_shape:torch.Size([1, 64, 496, 432])
        #>在原数据上变换维度,将一维伪图像又变成二维伪图像的形状，一个批次，496*432大小的伪图像，每个像素64个特征表示
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict