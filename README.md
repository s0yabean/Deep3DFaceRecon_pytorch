
## Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/colt18/Deep3DFaceRecon_pytorch.git
cd Deep3DFaceRecon_pytorch
conda env create -f environment.yml
source activate deep3dfacerecon
```

2. Install Nvdiffrast library:
```
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast    # ./Deep3DFaceRecon_pytorch/nvdiffrast
pip install .
```

3. Install Arcface Pytorch:
```
cd ..    # ./Deep3DFaceRecon_pytorch
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
```
## Inference with a pre-trained model

## Quickstart

Open the Colab: https://colab.research.google.com/drive/1CjLrVCKWPqeZ2oH9Bi5xeAiMUzMN_j3a#scrollTo=w2sKT7qnrDyI which is also the .ipynb file saved in this repo.

So far Tony would create the .obj file in Colab, save it locally and then run the remaining normalisation in local.

### Prepare prerequisite models

Link to 4 files: https://drive.google.com/drive/folders/1hRgefksqT6ZS7T6kAfjVC7zldob3dgLH?usp=drive_link

Copy drive directory to main folder and then:
```
cp drive/01_MorphableModel.mat ./BFM
cp drive/Exp_Pca.bin ./BFM

mkdir -p ./checkpoints/custom/
cp drive/epoch_20.pth ./checkpoints/custom

mkdir -p ./checkpoints/lm_model/
cp /content/drive/MyDrive/upwork/drive/68lm_detector.pb ./checkpoints/lm_model
```

### Test with custom images
To reconstruct 3d faces from test images, organize the test image folder as follows:
```
Deep3DFaceRecon_pytorch
│
└─── datasets
    │
    └─── <test_folder_name>

```
## Step 1

and then run `face_landmark_detection.py`:
```
python face_landmark_detection.py --image_folder=<test_folder_name> 
```
This will create `detections` folder under specified test_folder The \*.txt files are detected 5 facial landmarks with a shape of 5x2, and have the same name as the corresponding images..

## Step 2

Then, run the test script:
```
# get reconstruction results of your custom images
python test.py --name=custom --epoch=20 --img_folder=<folder_to_test_images>

# get reconstruction results of example images
python test.py --name=custom --epoch=20 --img_folder=./datasets/examples
```
## Mesh alignment code

This can be re-used for other 3D libraries as well, which manually rotates any 3D mesh into a preferred angle, and only works for meshes that have the same total length for different faces. For example, this library has 

It works by manually selecting the nose tip and middle of the 2 eyes right to get the indexes of the 3 points. To summarise, this code works for meshes where the total vertices is the same, no matter the face input used.

## Step 3

1. find_points.py

`find_points.py` script allows you to select points from a 3D mesh file (in OBJ format) using PyVista. It visualizes the 3D model, enabling the user to pick points interactively. The selected point IDs are stored in a list, and once three points are selected, the IDs are saved to a YAML file. The program then closes the visualization window. Later, saved points are used for rotation and translation.
```
python find_points.py <path/to/obj/file>
```

## Step 4

2. 3d_alignment.py

`3d_alignment.py` script processes a 3D face mesh and an associated image. It reads an OBJ file to load the 3D mesh and applies transformations based on user-defined rotation angles for the x, y, and z axes. It also locates and displays an image with the same base name as the OBJ file, if available. The transformed and rotated mesh is visualized alongside the original mesh using PyVista.

Usage:
use default values for transformation (40,0,0) for (x,y,z) axes.
```
python 3d_alignment.py <path/to/obj/file>
```
use keyword --rotation_angles for user defined rotation

```
python 3d_alignment.py <path/to/obj/file> --rotation_angles 30 30 -30
```
55 -1 0 seems to be the closest visually for most cases, not 40.

## Step 5

Based on a normalised rotation eg. (55, -1, 0) for deep3dface, we can proceed to obtain different angles of the face and save as images, which are then converted to embeddings for distance comparison. 