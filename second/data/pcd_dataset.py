import os
import pcl

from second.data.dataset import Dataset

class PCDDataset(Dataset):
    """
        To be complete.
    """

    def __init__(self, dir):
        self.dataset_dir = dir
        self.dataset_len = len(os.listdir(dir))
    
    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)

    
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
        filename = self.dataset_dir + '/' +str(query)+ ".pcd"
        points = pcl.load(filename)
        print(points.size)
        return res

    def create_PCD_info_file(self):
        pass

if __name__ == "__main__":
    dir = "/home/cm/Projects/deep_learning_projects/dataset/second_custom"
    dataset = PCDDataset(dir)
    print(dataset.get_sensor_data(3))