# SECOND for KITTI object detection

This is a ROS version for SECOND--a net for 3D Object detection. Origin ver is [here](https://github.com/traveller59/second.pytorch)

Thanks for [traveller59](https://github.com/traveller59)

## dependence

* [python-pcl](https://github.com/strawlab/python-pcl)

## usage

### PCD dataset.

Impelement custom datafile in pcd format.

1. rename pcd files into 0.pcd , 1.pcd, 2.pcd ....
2. in kittiviewer,set root info as pcd files dir.
3. click 'load'.
4. python ./data/pcd_dataset.py create_PCD_info_file --data_path=pcd_root_path
5. build net in kitti viewer and inference.

## TODO

- [ ] Predict single point cloud file.
- [ ] ROS msg to tensor.
- [x] Custom point cloud apply.
