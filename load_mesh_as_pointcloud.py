import argparse
import os
import numpy as np
import scipy.io as sio
import trimesh
from sklearn import preprocessing



def mkdir(dir):
    '''
    描述：文件夹不存在就创建
    参数：文件夹
    返回：无
    '''
    if not os.path.exists(dir):
        os.mkdir(dir)    

def sample_points(source_dir, target_dir, num_points):
    '''
    描述：对一个mesh进行采样保存成点云
    参数：mesh目录，target目录，点个数
    返回：无
    '''
    mesh = trimesh.load_mesh(source_dir)
    points, index = trimesh.sample.sample_surface(mesh, num_points)
    triangles = mesh.triangles[index]
    pt1 = triangles[:, 0, :]
    pt2 = triangles[:, 1, :]
    pt3 = triangles[:, 2, :]
    norm = np.cross(pt3 - pt1, pt2 - pt1)
    norm = preprocessing.normalize(norm, axis=1)
    sio.savemat(target_dir, {'v': np.array(points).astype(float), 'f': np.array(norm).astype(float)}, oned_as='row')

def load_data(args):
    '''
    描述：读取整个数据集
    参数：全局参数
    返回：无
    '''
    mkdir(args.target_base)
    for category in args.categories:
        target_dir = os.path.join(args.target_base, category)
        mkdir(target_dir)
        for use_type in ['train', 'val', 'test']:
            source_dir = os.path.join(args.source_base, use_type, category)
            objs = os.listdir(source_dir)
            for obj in objs:
                obj_model_path = os.path.join(source_dir, obj, 'model.ply')
                obj_result_path = os.path.join(target_dir, obj + '.mat')
                sample_points(obj_model_path, obj_result_path, args.num_points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_base', type = str, default = '/home/shenguanlin/ShapeFlow/data/shapenet_simplified')
    parser.add_argument('--target_base', type = str, default = '/home/shenguanlin/TMNet/data/customShapeNet_mat')
    parser.add_argument('--category', type = str, default = None)
    parser.add_argument('--num_points', type = int, default = 10000)
    args = parser.parse_args()
    if args.category == None:
        args.categories = ['02691156', '02958343', '03001627']
    else: 
        args.categories = [args.category]

    if not os.path.exists(args.source_base):
        raise Exception("wrong model path")
    load_data(args)
