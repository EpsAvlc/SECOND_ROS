from pathlib import Path
from google.protobuf import text_format
import torch
import pcl
import pickle
import numpy as np
from functools import partial

from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.data.pcd_dataset import PCDDataset
from second.pytorch.train import build_network, example_convert_to_torch
from second.data.preprocess import prep_pointcloud
from second.utils.config_tool import get_downsample_factor
from second.builder import dbsampler_builder

class pcd_inferencer():
    def __init__(self, config_path, checkpoint_path):
        self.dataset = PCDDataset
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.build_network()

        prep_cfg = self.config.preprocess
        dataset_cfg = self.config.dataset
        model_config = self.config.model.second
        num_point_features = model_config.num_point_features
        out_size_factor = get_downsample_factor(model_config)
        assert out_size_factor > 0
        cfg = self.config
        db_sampler_cfg = prep_cfg.database_sampler
        db_sampler = None
        if len(db_sampler_cfg.sample_groups) > 0:  # enable sample
            db_sampler = dbsampler_builder.build(db_sampler_cfg)
        grid_size = self.net.voxel_generator.grid_size
        # [352, 400]
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        print("feature_map_size", feature_map_size)
        assert all([n != '' for n in self.net.target_assigner.classes]), "you must specify class_name in anchor_generators."

        self.prep_func = partial(
            prep_pointcloud,
            root_path=dataset_cfg.kitti_root_path,
            voxel_generator=self.net.voxel_generator,
            target_assigner=self.net.target_assigner,
            training=False,
            max_voxels=prep_cfg.max_number_of_voxels,
            remove_outside_points=False,
            remove_unknown=prep_cfg.remove_unknown_examples,
            create_targets=False,
            shuffle_points=prep_cfg.shuffle_points,
            gt_rotation_noise=list(prep_cfg.groundtruth_rotation_uniform_noise),
            gt_loc_noise_std=list(prep_cfg.groundtruth_localization_noise_std),
            global_rotation_noise=list(prep_cfg.global_rotation_uniform_noise),
            global_scaling_noise=list(prep_cfg.global_scaling_uniform_noise),
            global_random_rot_range=list(
                prep_cfg.global_random_rotation_range_per_object),
            global_translate_noise_std=list(prep_cfg.global_translate_noise_std),
            db_sampler=db_sampler,
            num_point_features=3,
            anchor_area_threshold=prep_cfg.anchor_area_threshold,
            gt_points_drop=prep_cfg.groundtruth_points_drop_percentage,
            gt_drop_max_keep=prep_cfg.groundtruth_drop_max_keep_points,
            remove_points_after_sample=prep_cfg.remove_points_after_sample,
            remove_environment=prep_cfg.remove_environment,
            use_group_id=prep_cfg.use_group_id,
            out_size_factor=out_size_factor)

    def build_network(self):
        cfg_path = Path(self.config_path)
        ckpt_path = Path(self.checkpoint_path)
        if not cfg_path.exists():
            print("config file not exist.")
            return 
        if not ckpt_path.exists():
            print("ckpt file not exist.")
            return
        pass

        self.config = pipeline_pb2.TrainEvalPipelineConfig()

        with open(cfg_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(self.config.model.second).to(device).float().eval()
        if not ckpt_path.is_file():
            raise ValueError("checkpoint {} not exist.".format(ckpt_path))
        self.net.load_state_dict(torch.load(ckpt_path))
        
    def get_pc_data(self, pcd_path):
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
        }
        pc = pcl.load(pcd_path)
        points = []
        for point in pc:
            points.append([point[0], point[1], point[2]])
        # print(points)
        res["lidar"]["points"] = np.array(points)
        print(res["lidar"]["points"].shape)
        return res

    def infer(self, pcd_path):
        pc_data = self.get_pc_data(pcd_path)
        # prep_func = partial(prep_func, anchor_cache=anchor_cache)
    


if __name__ == "__main__":
    cfg_path = "/home/cm/Projects/deep_learning_projects/second.pytorch_v1.6/second/configs/car.fhd.config"
    ckpt_path = "/home/cm/Projects/deep_learning_projects/second.pytorch_v1.6/second/models/voxelnet-15475.tckpt"
    infer = pcd_inferencer(cfg_path, ckpt_path)

