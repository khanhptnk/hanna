### 1. Download ANNA dataset:

```
$ bash download.sh
```

The following directories will be created:
* `hanna`: HANNA dataset. Please read our paper for more details.
* `connectivity`: environment graphs.
* `img_features`: precomputed image embeddings computed by ResNet-152 (provided by the [Matterport3D simulator](https://github.com/peteanderson80/Matterport3DSimulator)). 

### 2. Download Matterport3D dataset

Request access to the dataset [here](https://niessner.github.io/Matterport/). The dataset is for **non-commercial academic purposes** only. Please read and agree to the dataset's [terms and conditions](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and **put a link to them in your project repo** (as requested by the dataset's creators).

You need to download the `matterport_skybox_images`, `undistorted_camera_parameters`, and `house_segmentations` portions and unzip the files so that `$MATTERPORT_DATA_DIR/<scanId>/matterport_skybox_images/*.jpg` and `$MATTERPORT_DATA_DIR/<scanId>/undistorted_camera_parameters/*.conf` `$MATTERPORT_DATA_DIR/<scanId>/house_segmentations/panorama_to_region.txt` files are present. 

We provide the script `unzip_matterport.sh` in this directory to help you unzip the data:
```
$ export MATTERPORT_DATA_DIR=<some_path>/v1/scans
$ bash unzip_matterport.sh $MATTERPORT_DATA_DIR $file_type
```
where `$file_type` is a file type (e.g., `matterport_skybox_images` or `undistorted_camera_parameters`).

**Next: [Setup simulator](https://github.com/khanhptnk/hanna-private/tree/master/code)**


