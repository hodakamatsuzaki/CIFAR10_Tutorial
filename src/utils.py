import os
import json
import pickle

def readConfig(path:str)->dict:
    json_file = open(path, 'r')
    json_dict = json.load(json_file)
    return json_dict

def makeDirs(output_path:str, sub_dir_list=None)->None:
    if sub_dir_list is not None:
        for tmp_dir in sub_dir_list:
            os.makedirs(f'{output_path}/{tmp_dir}', exist_ok=True)
    else:
        os.makedirs(output_path, exist_ok=True)
    return None

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data