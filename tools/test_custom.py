

import argparse
import glob
from pathlib import Path
from unittest.mock import DEFAULT

from sympy import EX, Integer
from traitlets import default

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import time
import redis

#全局常量定义

#默认配置文件路径(可由命令行参数指定)
DEFAULT_CONFIG_FILE="/home/tangh/workspace/dengyuhan/kpp/OpenPCDet/tools/cfgs/kitti_models/pointpillar_custom.yaml"
#默认测试集路径(可由命令行参数指定)
DEFAULT_DATA_PATH="/home/tangh/workspace/dengyuhan/kpp/grab_bucket_test_data/"
#默认预训练权重文件路径(可由命令行参数指定)
DEFAULT_CKPT="/home/tangh/workspace/dengyuhan/kpp/OpenPCDet/output/kitti_models/pointpillar_custom/default/ckpt/checkpoint_epoch_160.pth"
#默认执行动作(可由命令行参数指定)
DEFAULT_ACTION="print_average_score"

#解析命令行参数和模型配置文件
def parse_config():
    #创建解析器
    parser = argparse.ArgumentParser(description='arg parser')
    #添加参数:模型配置文件
    parser.add_argument('--cfg_file', 
                        type=str, 
                        default=DEFAULT_CONFIG_FILE,
                        help='specify the config for model')
    #添加参数:数据路径
    parser.add_argument('--data_path',
                        type=str, 
                        default=DEFAULT_DATA_PATH,
                        help='specify the point cloud data file or directory')
    #添加参数:预训练权重文件
    parser.add_argument('--ckpt', 
                        type=str, 
                        default=DEFAULT_CKPT, 
                        help='specify the pretrained model')
    #添加参数:脚本执行动作
    parser.add_argument('--action',
                        type=str,
                        default=DEFAULT_ACTION,
                        help='specify script actions: print_results, view_results, send_results')
    #解析命令行获取参数集合
    args = parser.parse_args()
    #解析模型配置文件
    cfg_from_yaml_file(args.cfg_file, cfg)
    #返回参数集合和模型配置
    return args, cfg

#命令行参数和模型配置
COMMAND_ARGS,MODEL_CONFIG=parse_config()

#基于目录或文件的数据加载器
class FileDatasetLoader(DatasetTemplate):
    #初始化
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        #初始化父类
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        #设置测试集路径
        self.root_path = root_path
        #设置扩展名
        self.ext = ext
        #获取测试集数据文件列表
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        #数据文件列表排序
        data_file_list.sort()
        #设置数据文件列表
        self.sample_file_list = data_file_list

    #回调:获取样本大小
    def __len__(self):
        #返回测试集文件数
        return len(self.sample_file_list)

    #回调:根据索引获取点云数据
    def __getitem__(self, index):
        if self.ext == '.bin':
            #读取bin格式的点云数据
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 3)
        elif self.ext == '.npy':
            #读取npy格式的点云数据
            points = np.load(self.sample_file_list[index])
        else:
            #非法格式
            raise NotImplementedError

        #将点云数据封装成字典
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        #转换为数据字典
        data_dict = self.prepare_data(data_dict=input_dict)
        #返回包含一帧点云数据的数据字典
        return data_dict



#创建并初始化模型
def init_model():
    #创建日志器
    logger = common_utils.create_logger()
    #数据加载器为文件数据加载器
    dataset = FileDatasetLoader(
        dataset_cfg=MODEL_CONFIG.DATA_CONFIG, class_names=MODEL_CONFIG.CLASS_NAMES, training=False,
        root_path=Path(COMMAND_ARGS.data_path), logger=logger)
    #创建模型
    model = build_network(model_cfg=MODEL_CONFIG.MODEL, num_class=len(MODEL_CONFIG.CLASS_NAMES), dataset=dataset)
    #加载预训练权重文件
    model.load_params_from_file(filename=COMMAND_ARGS.ckpt, logger=logger, to_cpu=True)
    #初始化cuda
    model.cuda()
    #设置为推理模式
    model.eval()
    #返回模型和数据加载器
    return model,dataset

#检测点云数据
def detect(model,dataset,data_dict):
    #调用数据加载器回调方法获取一帧数据字典
    data_dict = dataset.collate_batch([data_dict])
    #数据转移至GPU
    load_data_to_gpu(data_dict)
    #模型前向推理获得预测结果
    pred_dicts, _ = model.forward(data_dict)
    #返回预测结果
    return pred_dicts

#检测点云并打印结果
def print_results():
    #获取模型和数据加载器
    model,dataset=init_model()
    #获取总样本数量
    total_sample_num=len(dataset)
    #记录包含检测框的样本数量
    boxed_sample_num=0
    with torch.no_grad():
        #遍历测试集
        for index, data_dict in enumerate(dataset):
            #获取推理开始时间
            start_time=time.time()
            #前向推理得到推理结果
            pred_dicts=detect(model,dataset,data_dict)
            #获取推理结束时间
            end_time=time.time()

            #打印点云数据索引
            print("\n<<<点云数据索引 "+str(index)+" 检测框信息>>>\n")

            if pred_dicts[0]['pred_boxes'].shape[0]==1:
                #当前帧点云数据上有且只有一个检测框
                #打印检测结果
                print("检测框信息:",pred_dicts[0]['pred_boxes'][0])
                print("检测框评分:",pred_dicts[0]['pred_scores'][0])
                #计数增加
                boxed_sample_num+=1
            else:
                #当前帧点云数据上没有检测框
                #打印检测结果
                print("检测框信息:","None")
                print("检测框评分:","None")

            #打印当前帧点云数据推理耗时
            print("推理耗时(秒):",end_time-start_time)

    #打印推理结果总体概要信息
    print("\n测试集总样本数:",total_sample_num)
    print("含检测框样本数:",boxed_sample_num)

#检测点云并可视化结果
def view_results():
    #获取模型和数据加载器
    model,dataset=init_model()
    with torch.no_grad():
        #遍历测试集
        for index, data_dict in enumerate(dataset):
            #前向推理得到推理结果
            pred_dicts=detect(model,dataset,data_dict)
            #打印点云数据索引
            print("当前可视化点云数据索引:",index)
            #推理结果可视化
            V.draw_scenes(
                points=data_dict['points'], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

#打印平均分数
def print_average_score():
    #获取模型和数据加载器
    model,dataset=init_model()
    #获取总样本数量
    total_sample_num=len(dataset)
    #评分列表
    score_list=[]
    #推理时间列表
    time_list=[]
    with torch.no_grad():
        #遍历测试集
        for index, data_dict in enumerate(dataset):
            #获取推理开始时间
            start_time=time.time()
            #前向推理得到推理结果
            pred_dicts=detect(model,dataset,data_dict)
            #获取推理结束时间
            end_time=time.time()
            #统计当前样本检测框得分
            score_list.append(pred_dicts[0]['pred_scores'][0])
            #统计当前样本推理耗时
            time_list.append(end_time-start_time)

    #打印推理结果总体概要信息
    print("测试样本总数:",total_sample_num)
    print("已检测样本数:",len(score_list))
    print("检测框平均得分:",sum([score.item() for score in score_list])/len(score_list))
    print("平均检测时间:",sum(time_list)/len(time_list))
    print("检测框得分明细:",[score.item() for score in score_list])
    print("推理时间明细:",time_list)


#执行动作
def run():
    #执行指定动作
    if COMMAND_ARGS.action=="print_results":
        #执行打印结果动作
        print_results()
    elif COMMAND_ARGS.action=="view_results":
        #执行可视化结果动作
        view_results()
    elif COMMAND_ARGS.action=="print_average_score":
        #执行打印检测框平均得分动作
        print_average_score()
    else:
        #无法识别动作
        raise Exception("the action cannot be recognized")

if __name__ == '__main__':
    run()