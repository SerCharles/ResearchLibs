"""ShapeNet deformation dataloader"""
import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import trimesh
import glob
import warnings
import imageio
import pickle
from collections import OrderedDict
from scipy.spatial import cKDTree



def sample_mesh(mesh_path, nsamples, normals=True):
    """Load the mesh from mesh_path and sample nsampels points from its vertices.
        
    If nsamples < number of vertices on mesh, randomly repeat some vertices as padding.
        
    Args:
      mesh_path: str, path to load the mesh from.
      nsamples: int, number of vertices to sample.
      normals: bool, whether to add normals to the point features.
    Returns:
      v_sample: np array of shape [nsamples, 3 or 6] for sampled points.
    """
    mesh = trimesh.load(mesh_path)
    v = np.array(mesh.vertices)
    nv = v.shape[0]
    seq = np.random.permutation(nv)[:nsamples]
    if len(seq) < nsamples:
        seq_repeat = np.random.choice(nv, nsamples-len(seq), replace=True)
        seq = np.concatenate([seq, seq_repeat], axis=0)
    v_sample = v[seq]
    if normals:
        n_sample = np.array(mesh.vertex_normals[seq])
        v_sample = np.concatenate([v_sample, n_sample], axis=-1)
    
    return v_sample
    
def load_datas(args):
    '''
    描述：读取配对的点云列表
    参数：全局参数
    返回：一个列表，每个元素是配对的点云
    '''
    result_list = []
    base_dir_source = '/home/shenguanlin/TMNet/results'
    base_dir_source = os.path.join(base_dir_source, args.run_name)
    #base_dir_source = os.path.join(base_dir_source, 'result')
    base_dir_target = '/home2/lbq/deformmulti_changecode/data/shapenet_simplified/test' 
    categories = []
    if args.category == None:
        categories = ['02691156', '02958343', '03001627']
    else:
        categories = [args.category]
    for category in categories:
        category_dir_source = os.path.join(base_dir_source, category)
        category_dir_target = os.path.join(base_dir_target, category)
        filename_list = os.listdir(category_dir_source)
        for filename in filename_list:
            dir_source = os.path.join(category_dir_source, filename)
            dir_target = os.path.join(category_dir_target, filename)
            if not (os.path.exists(dir_source) and os.path.exists(dir_target)):
                continue
            filename_source = os.path.join(dir_source, 'model.ply') 
            filename_target = os.path.join(dir_target, 'model.ply')
            try:
                pointcloud_source = sample_mesh(filename_source, args.nsamples, args.normals)
                pointcloud_target = sample_mesh(filename_target, args.nsamples, args.normals)
            except:
                continue
            pointcloud_pair = pointcloud_source, pointcloud_target
            result_list.append(pointcloud_pair)
    return result_list

