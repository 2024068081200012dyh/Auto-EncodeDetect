import os
import numpy as np
import open3d as o3d

def pcd2bin(input_folder, output_folder):
    # 获取文件夹中所有的pcd文件
    pcd_files = [f for f in os.listdir(input_folder) if f.endswith('.pcd')]

    for pcd_file in pcd_files:
        pcd_path = os.path.join(input_folder, pcd_file)
        bin_file = pcd_file.replace('.pcd', '.bin')
        bin_path = os.path.join(output_folder, bin_file)

        # 读取PCD文件
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.array(pcd.points)

        binary_data = points.astype(np.float32).tobytes()
        with open(bin_path, 'wb') as f:
            f.write(binary_data)

        print(f"Converted {pcd_file} to {bin_file}")

if __name__=="__main__":
    input_folder = '/home/tangh/workspace/luolinfeng/Project/PCD_MMDet3D/data/kitti/training/velodyne/pcd'
    output_folder = '/home/tangh/workspace/luolinfeng/Project/PCD_MMDet3D/data/kitti/training/velodyne/bin'
    pcd2bin(input_folder, output_folder)
