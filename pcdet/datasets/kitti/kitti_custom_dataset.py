import copy
import pickle
import os

import numpy as np
from skimage import io

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, object3d_custom
from ..dataset import DatasetTemplate
# 定义属于自己的数据集，集成数据集模板
class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = os.path.join(self.root_path, ('training' if self.split != 'test' else 'testing'))

        split_dir = os.path.join(self.root_path, 'ImageSets',(self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.custom_infos = []
        self.include_custom_data(self.mode)
        self.ext = ext

    # 用于导入自定义数据
    def include_custom_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Custom dataset.')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)
        
        self.custom_infos.extend(custom_infos)

        if self.logger is not None:
            self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))
    
    # 用于获取标签的标注信息
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        # 线程函数，主要是为了多线程读取数据，加快处理速度
        
        # 处理一帧
        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            # 创建一个用于存储一帧信息的空字典
            info = {}
            # 定义该帧点云信息，pointcloud_info
            pc_info = {'num_features': 3, 'lidar_idx': sample_idx}
            # 将pc_info这个字典作为info字典里的一个键值对的值，其键名为‘point_cloud’添加到info里去
            info['point_cloud'] = pc_info
            if has_label:
                # 通过get_label函数，读取出该帧的标签标注信息
                obj_list = self.get_label(sample_idx)
                # 创建用于存储该帧标注信息的空字典
                annotations = {}
                # 下方根据标注文件里的属性将对应的信息加入到annotations的键值对，可以根据自己的需求取舍
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                # annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                # annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                # annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                # annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                # annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
 
                # 统计有效物体的个数，即去掉类别名称为“Dontcare”以外的
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                # 统计物体的总个数，包括了Dontcare
                num_gt = len(annotations['name'])
                # 获得当前的index信息
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
 
                # 从annotations里提取出从标注信息里获取的location、dims、rots等信息，赋值给对应的变量
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # 由于我们的数据集本来就是基于雷达坐标系标注，所以无需坐标转换
                #loc_lidar = calib.rect_to_lidar(loc)
                loc_lidar = self.get_calib(loc)
                # 原来的dims排序是高宽长hwl,现在转到pcdet的统一坐标系下,按lhw排布
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                
                # 由于我们基于雷达坐标系标注，所以获取的中心点本来就是空间中心，所以无需从底面中心转到空间中心
                # bottom center -> object center: no need for loc_lidar[:, 2] += h[:, 0] / 2
                # print("sample_idx: ", sample_idx, "loc: ", loc, "loc_lidar: " , sample_idx, loc_lidar)
                # get gt_boxes_lidar see https://zhuanlan.zhihu.com/p/152120636
                # loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                # 将雷达坐标系下的真值框信息存入annotations中
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                # 将annotations这整个字典作为info字典里的一个键值对的值
                info['annos'] = annotations
            
            return info
            # 后续的由于没有calib信息和image信息，所以可以直接注释
            '''
            #     if count_inside_pts:
            #         points = self.get_lidar(sample_idx)
            #         calib = self.get_calib(sample_idx)
            #         pts_rect = calib.lidar_to_rect(points[:, 0:3])
            #         fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
            #         pts_fov = points[fov_flag]
            #         corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
            #         num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
            #         for k in range(num_objects):
            #             flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
            #             num_points_in_gt[k] = flag.sum()
            #         annotations['num_points_in_gt'] = num_points_in_gt
            # return info
            '''
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
        # 此时返回值infos是列表，列表元素为字典类型
                
    # 用于获取标定信息
    def get_calib(self, loc):
        # calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        # assert calib_file.exists()
        # return calibration_kitti.Calibration(calib_file)
        
        # loc_lidar = np.concatenate([np.array((float(loc_obj[2]),float(-loc_obj[0]),float(loc_obj[1]-2.3)),dtype=np.float32).reshape(1,3) for loc_obj in loc])
        # return loc_lidar
        # 这里做了一个由相机坐标系到雷达坐标系翻转（都遵从右手坐标系），但是 -2.3这个数值具体如何得来需要再看下
 
        # 我们的label中的xyz就是在雷达坐标系下,不用转变,直接赋值
        loc_lidar = np.concatenate([np.array((float(loc_obj[0]),float(loc_obj[1]),float(loc_obj[2])),dtype=np.float32).reshape(1,3) for loc_obj in loc])
        return loc_lidar
                
    # 用于获取标签
    def get_label(self, idx):
        # 从指定路径中提取txt内容
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        # 主要就是从这个函数里获取具体的信息
        return object3d_custom.get_objects_from_label(label_file)

    # 用于获取雷达点云信息
    def get_lidar(self, idx, getitem):
        """
            Loads point clouds for a sample
                Args:
                    index (int): Index of the point cloud file to get.
                Returns:
                    np.array(N, 4): point cloud.
        """
        # get lidar statistics
        if getitem == True:
            lidar_file = self.root_split_path + '/velodyne/' + ('%s.bin' % idx)
        else:
            lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 3)

    # 用于数据集划分
    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    # 创建真值数据库
    # Create gt database for data augmentation
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
            import torch
    
            database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
            db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)
    
            database_save_path.mkdir(parents=True, exist_ok=True)
            all_db_infos = {}
    
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
    
            for k in range(len(infos)):
                print('gt_database sample: %d/%d' % (k + 1, len(infos)))
                info = infos[k]
                sample_idx = info['point_cloud']['lidar_idx']
                points = self.get_lidar(sample_idx,False)
                annos = info['annos']
                names = annos['name']
                # difficulty = annos['difficulty']
                # bbox = annos['bbox']
                gt_boxes = annos['gt_boxes_lidar']
    
                num_obj = gt_boxes.shape[0]
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)
    
                for i in range(num_obj):
                    filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[point_indices[i] > 0]
    
                    gt_points[:, :3] -= gt_boxes[i, :3]
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)
    
                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        # db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                        #            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                        #            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                        db_info = {'name': names[i], 'path': db_path,  'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0], 'score': annos['score'][i]}
                        
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
            for k, v in all_db_infos.items():
                print('Database %s: %d' % (k, len(v)))
    
            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)
    # 生成预测字典信息
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N,7), Tensor
                pred_scores: (N), Tensor
                pred_lables: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_smaples):
            ret_dict = {
                'name': np.zeros(num_smaples), 'alpha' : np.zeros(num_smaples),
                'dimensions': np.zeros([num_smaples, 3]), 'location': np.zeros([num_smaples, 3]),
                'rotation_y': np.zeros(num_smaples), 'score': np.zeros(num_smaples),
                'boxes_lidar': np.zeros([num_smaples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            # Define an empty template dict to store the prediction information, 'pred_scores.shape[0]' means 'num_samples'
            pred_dict = get_template_prediction(pred_scores.shape[0])
            # If num_samples equals zero then return the empty dict
            if pred_scores.shape[0] == 0:
                return pred_dict

            # No calibration files

            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes,None)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            # pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            # pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            # pred_dict['location'] = pred_boxes_camera[:, 0:3]
            # pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            # Output pred results to Output-path in .txt file 
            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl: lidar -> camera

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                            % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                single_pred_dict['score'][idx]), file=f)
            return annos
    def evaluation(self, det_annos, class_names, **kwargs):
            if 'annos' not in self.custom_infos[0].keys():
                return None, {}

            # from .kitti_object_eval_python import eval as kitti_eval
            from .kitti_object_eval_python import eval_custom as kitti_eval

            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

            return ap_result_str, ap_dict
    # 用于返回训练帧的总个数
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_infos)

    # 用于将点云与3D标注框均转至前述统一坐标定义下，送入数据基类提供的self.prepare_data()
    def __getitem__(self, index):  ## 修改如下
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx, True)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

# 用于创建自定义数据集的信息
def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'
   # 定义文件的路径和名称
    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'custom_infos_trainval.pkl'
    test_filename = save_path / 'custom_infos_test.pkl'
 
    print('---------------Start to generate data infos---------------')
 
    dataset.set_split(train_split)
    # 执行完上一步，得到train相关的保存文件，以及sample_id_list的值为train.txt文件下的数字
    # 下面是得到train.txt中序列相关的所有点云数据的信息，并且进行保存
    custom_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)
 
    dataset.set_split(val_split)
    # 对验证集的数据进行信息统计并保存
    custom_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info val file is saved to %s' % val_filename)
 
    with open(trainval_filename, 'wb') as f:
        pickle.dump(custom_infos_train + custom_infos_val, f)
    print('Custom info trainval file is saved to %s' % trainval_filename)
 
 
    dataset.set_split('test')
    # kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    custom_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(custom_infos_test, f)
    print('Custom info test file is saved to %s' % test_filename)
 
    
 
    print('---------------Start create groundtruth database for data augmentation---------------')
    # 用trainfile产生groundtruth_database
    # 只保存训练数据中的gt_box及其包围点的信息，用于数据增强    
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(info_path=train_filename, split=train_split)
 
    print('---------------Data preparation Done---------------')

if __name__=='__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['GrabBucket'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
