import argparse
import trimesh
from get_iou import iou_mesh
from data import load_datas

def init_args():
    parser = argparse.ArgumentParser(description = "Get IOU")
    parser.add_argument("--category", type = str, default = None, help = "The name of categories")
    parser.add_argument("--run_name", type = str, default = 'all', help = "The name of run")
    args = parser.parse_args()
    if args.category == None:
        args.run_name == 'all'
    else: 
        args.run_name = args.category
    return args

if __name__ == "__main__":
    args = init_args()
    print(args)
    data_list = load_datas(args)
    print("Data loaded")

    total_iou = 0.0
    for pair in data_list:
        source, target = pair

        iou = iou_mesh(source, target)
        total_iou += iou
        print("IOU: {:.6f}".format(iou))
    total_iou /= len(data_list)
    print("Average IOU is {:.6f}".format(total_iou))

