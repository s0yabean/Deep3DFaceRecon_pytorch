
## Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/colt18/Deep3DFaceRecon_pytorch.git
cd Deep3DFaceRecon_pytorch
conda env create -f environment.yml
source activate deep3d_pytorch
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

### Prepare prerequisite models
Copy drive directory to main folder and then:
```
cp drive/01_MorphableModel.mat ./BFM
cp drive/Exp_Pca.bin ./BFM
mkdir -p ./checkpoints/custom/
cp drive/epoch_20.pth ./checkpoints/custom
```

### Test with custom images
To reconstruct 3d faces from test images, organize the test image folder as follows:
```
Deep3DFaceRecon_pytorch
│
└─── <folder_to_test_images>
    │
    └─── *.jpg/*.png
    |
    └─── detections
        |
	└─── *.txt
```
The \*.jpg/\*.png files are test images. The \*.txt files are detected 5 facial landmarks with a shape of 5x2, and have the same name as the corresponding images. Check [./datasets/examples](datasets/examples) for a reference.

Then, run the test script:
```
# get reconstruction results of your custom images
python test.py --name=custom --epoch=20 --img_folder=<folder_to_test_images>

# get reconstruction results of example images
python test.py --name=custom --epoch=20 --img_folder=./datasets/examples
```
## Mesh alignment code

1. find_points.py

`find_points.py` script allows you to select points from a 3D mesh file (in OBJ format) using PyVista. It visualizes the 3D model, enabling the user to pick points interactively. The selected point IDs are stored in a list, and once three points are selected, the IDs are saved to a YAML file. The program then closes the visualization window. Later, saved points are used for rotation and translation.
```
python find_points.py <path/to/obj/file>
```

2. 3d_alignment.py

`3d_alignment.py` script processes a 3D face mesh and an associated image. It reads an OBJ file to load the 3D mesh and applies transformations based on user-defined rotation angles for the x, y, and z axes. It also locates and displays an image with the same base name as the OBJ file, if available. The transformed and rotated mesh is visualized alongside the original mesh using PyVista.

Usage:
use default values for transformation (40,0,0) for (x,y,z) axes.
 ```
 python 3d_alignment.py checkpoints/custom/results/examples/epoch_20_000000/000006.obj
 ```
 use keyword --rotation_angles for user defined rotation
```
python 3d_alignment.py checkpoints/custom/results/examples/epoch_20_000000/000006.obj --rotation_angles 30 30 -30
```
