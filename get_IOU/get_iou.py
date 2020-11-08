import numpy as np 
import os
import trimesh
import trimesh.voxel.creation

def iou_mesh(mesh1, mesh2):
    voxel1 = trimesh.voxel.creation.local_voxelize(mesh1,(0,0,0),0.04,16)
    voxel1 = voxel1.matrix.transpose(0,2,1)
    voxel2 = trimesh.voxel.creation.local_voxelize(mesh2,(0,0,0),0.04,16)
    voxel2 = voxel2.matrix.transpose(0,2,1)
    insection = 0
    union = 0
    for i in range(len(voxel1)):
        for j in range(len(voxel1[i])):
            for k in range(len(voxel1[i, j])):
                if voxel1[i, j, k] == True and voxel2[i, j, k] == True:
                    insection += 1
                if voxel1[i, j, k] == True or voxel2[i, j, k] == True:
                    union += 1
    return insection / union
