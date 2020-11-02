import os
import json


def gether_data():
    data = {}
    data_base = os.path.join('data', 'shapenet_simplified')
    data_base_train  = os.path.join(data_base, 'train')    
    data_base_val  = os.path.join(data_base, 'val')
    data_base_test  = os.path.join(data_base, 'test')
    typecode = os.listdir(data_base_train)
    for the_type in typecode:
        data[the_type] = {}
        dir_train = os.path.join(data_base_train, the_type)
        dir_val = os.path.join(data_base_val, the_type)
        dir_test = os.path.join(data_base_test, the_type)
        file_train = os.listdir(dir_train)
        file_val = os.listdir(dir_val)
        file_test = os.listdir(dir_test)
        data[the_type]['train'] = file_train
        data[the_type]['val'] = file_val
        data[the_type]['test'] = file_test
    
    for the_type in typecode:
        print(the_type, 'train', len(data[the_type]['train']))
        print(the_type, 'val', len(data[the_type]['val']))
        print(the_type, 'test', len(data[the_type]['test']))

    result = json.dumps(data)
    f = open('file_statistics.json', 'w')
    f.write(result)
    f.close()

gether_data()