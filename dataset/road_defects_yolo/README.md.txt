#RoadDefects-ISeg Dataset

##Overview:
RoadDefects-ISeg is a dataset for instance segmentation of road surface defects..  
It contains 1000 images across 4 classes (250 images each):
- Crack
- Lane
- Pothole
- Speed_Breaker
The dataset is split into train (80%), validation (10%), and test (10%) subsets.

##Dataset Structure:
RoadDefects-ISeg/
│
├── data.yaml              # YOLO dataset configuration
├── README.md              # Dataset description
├── LICENSE                # CC BY 4.0 license
│
├── train/
│   ├── images/            # Training images
│   └── labels/            # YOLO-format segmentation labels
│
├── val/
│   ├── images/            # Validation images
│   └── labels/
│
└── test/
    ├── images/            # Test images
    └── labels/

images/ → Input images
labels/ → Instance segmentation annotations in YOLO format
data.yaml → Dataset configuration file for YOLO training  

##Notes:
-The original datasets included a mix of object detection and segmentation annotations, which could not be directly used. All images were manually re-annotated to provide precise instance segmentation labels suitable for this project.
-This dataset is intended for research and academic use only.

##Acknowledgments:
We thank the authors of the original datasets for making their data available under the CC BY 4.0 license, which enabled the creation of this dataset:
1. My Road Crack Dataset – Roboflow Universe (https://universe.roboflow.com/newroadcrack/my-road-crack-dataset/dataset/7)
2. Speedbreaker Final Dataset – Roboflow Universe(https://universe.roboflow.com/cse400/speedbreaker_final/dataset/4)
3. Pothole Segmentation for Road Damage Assessment – Kaggle(https://www.kaggle.com/code/farzadnekouei/pothole-segmentation-for-road-damage-assessment/input)
4. Lane Dataset – Roboflow Universe(https://universe.roboflow.com/driveguard/lane-z4jgy-zh5j0/dataset/1)