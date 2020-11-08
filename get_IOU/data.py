import os
import numpy as np
import trimesh


def load_datas(args):
    '''
    描述：读取配对的面片列表
    参数：全局参数
    返回：一个列表，每个元素是配对的面片
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
        #print(category_dir_source)
        filename_list = os.listdir(category_dir_source)
        for filename in filename_list:
            dir_source = os.path.join(category_dir_source, filename)
            dir_target = os.path.join(category_dir_target, filename)
            if not (os.path.exists(dir_source) and os.path.exists(dir_target)):
                continue
            filename_source = os.path.join(dir_source, 'model.ply') 
            filename_target = os.path.join(dir_target, 'model.ply')
            try:
                mesh_source = trimesh.load(filename_source)
                mesh_target = trimesh.load(filename_target)
            except:
                continue
            mesh_pair = mesh_source, mesh_target
            result_list.append(mesh_pair)
    return result_list