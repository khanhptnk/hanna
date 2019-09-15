### 1. Download ANNA dataset:

```
$ bash download.sh
```

The following directories will be created:
* `anna`: ANNA dataset. Please read our paper for more details.
* `connectivity`: environment graphs.
* `img_features`: precomputed image embeddings computed by ResNet (pretrained on ImageNet). 

### 2. Download Matterport3D dataset

Request access to the dataset [here](https://niessner.github.io/Matterport/). The dataset is for **non-commercial academic purposes** only. Please read and agree to the dataset's [terms and conditions](http://dovahkiin.stanford.edu/matterport/public/MP_TOS.pdf) and **put a link to them in your project repo** (as requested by the dataset's creators).

Training and testing our models only require downloading the `house_segmentations` portion of the dataset. Unzip the files so that `<some_folder>/v1/scans/<scanId>/house_segmentations/panorama_to_region.txt` are present. 

Running in graphics mode is still useful for debugging and visualizing the agent behavior. You need to download the `matterport_skybox_images` and `undistorted_camera_parameters` portions and unzip the files so that `$MATTERPORT_DATA_DIR/<scanId>/matterport_skybox_images/*.jpg` and `$MATTERPORT_DATA_DIR/<scanId>/undistorted_camera_parameters/*.conf` files are present. 

We provide the script `unzip_matterport.sh` in this directory to help you unzip the data:
```
$ bash unzip_matterport $MATTERPORT_DATA_DIR $file_type
```
where `$file_type` is a file type (e.g., `matterport_skybox_images` or `undistorted_camera_parameters`).

Next: [Setup simulators](https://github.com/debadeepta/learningtoask/tree/master/code)

