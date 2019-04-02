import os
import pcl
import numpy as np
from pathlib import Path
import concurrent.futures as futures
import pickle
import fire

from second.data.dataset import Dataset

class PCDDataset(Dataset):
    """
        To be complete.
    """

    NumPointFeatures = 3
    def __init__(self, root_path, info_path=None, class_names=None,
                 prep_func=None,
                 num_point_features=None):
        self.dataset_dir = root_path
        self.dataset_len = len(os.listdir(root_path))
        self._prep_func=prep_func
    
    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        return example
    
    def __len__(self):
        return self.dataset_len
    
    def get_sensor_data(self, query):
        assert isinstance(query, int)
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
        }
        print(query)
        filename = str(self.dataset_dir) + '/' +str(query)+ ".pcd"
        pc = pcl.load(filename)
        points = []
        for point in pc:
            points.append([point[0], point[1], point[2]])
        # print(points)
        res["lidar"]["points"] = np.array(points)
        print(res["lidar"]["points"].shape)
        return res

def get_PCD_path(idx, prefix):
    return str(prefix + (str(idx) + ".pcd")) 

def get_PCD_info(path,
                training=True,
                label_info=True,
                velodyne=False,
                calib=False,
                image_ids=7481,
                extend_matrix=True,
                num_worker=8,
                relative_path=True,
                with_imageshape=True):
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids)) 
    
    def map_func(idx):
        info = {}
        pc_info = {'num_features': 3}
        calib_info = {}
        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_PCD_path(
                idx, path)
        info["image"] = image_info
        info["point_cloud"] = pc_info
        return info
    
    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)

def create_PCD_info_file(data_path, save_path=None, relative_path=True):
    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    PCD_infos_test = get_PCD_info(
            data_path,
            training=False,
            label_info=False,
            velodyne=True,
            calib=False,
            image_ids=len(os.listdir(data_path)),
            relative_path=relative_path)
    filename = save_path / 'pcd_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(PCD_infos_test, f)

if __name__ == "__main__":
    # dir = "/home/cm/Projects/deep_learning_projects/dataset/second_custom"
    # dataset = PCDDataset(dir)
    # print(dataset.get_sensor_data(3))
    fire.Fire()