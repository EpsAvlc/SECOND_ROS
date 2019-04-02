from pathlib import Path
from google.protobuf import text_format
import torch
import pcl
import pickle

from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.data.pcd_dataset import PCDDataset
from second.pytorch.train import build_network, example_convert_to_torch

class pcd_inferencer():
    def __init__(self):
        self.dataset = PCDDataset

    def build_network(self, config_path, checkpoint_path):
        cfg_path = Path(config_path)
        ckpt_path = Path(config_path)
        if not cfg_path.exists():
            print("config file not exist.")
            return 
        if not ckpt_path.exists():
            print("ckpt file not exist.")
            return
        pass

        config = pipeline_pb2.TrainEvalPipelineConfig()

        with open(cfg_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        
        device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(config.model.second).to(device).float().eval()
        self.net.load_state_dict(torch.load(ckpt_path))
        
    
    def inference(self, pcd_path):
        
        pass

if __name__ == "__main__":
    cfg_path = "/home/cm/Projects/deep_learning_projects/second.pytorch_v1.6/second/configs/car.fhd.config"
    ckpt_path = "/home/cm/Projects/deep_learning_projects/second.pytorch_v1.6/second/models/voxelnet-30950.tckpt"

    infer = pcd_inferencer()
    infer.build_network(cfg_path, ckpt_path)
