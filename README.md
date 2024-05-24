# 3D-Geological-Modeling
This repo is for 3D geological or geotechnical modelling based on machine learning

## Models
- [x] K Nearest Neighbour (kNN)
- [x] Support Vector Machine (SVM)
- [x] Random Forest (RF)
- [x] Gradient Boosting Classifiers (GBC)
- [x] Fully Connected Neural Network (NN)
- [x] Ensemble Method
- [x] Indicator Kriging 3D

## Directory Structure
```ascii
.
├── Borehole reading /
│   ├── AGS Lab processing.py
│   ├── AGS Processing.py
│   ├── AGS Reading.py
│   ├── BH Error Check.py
│   ├── CPT Processing.py
│   └── Borehole SPT CPT data processing.txt
├── 3D Modelling/
│   ├── Geological modelling/
│   |   ├── 3D by kNN.py
│   |   ├── 3D by RF.py
│   |   ├── 3D by Stacked.py
│   |   ├── 3D by SVM.py
│   |   ├── 3D for whole HK.py
│   |   ├── 3D GBC.py
│   |   ├── 3D NN.py
│   |   ├── IndicatorKriging3D.py
│   |   └── remove land.txt
│   ├── Geotechnical modelling/
│   |   ├── 3D Geotechnical Model by kNN.py
│   |   ├── Check Soil Materials over SPT.py
│   |   ├── CPT Model.py
│   |   ├── CPT sheet separation.py
│   |   ├── CPT-SPT relation.py
│   |   ├── Geo SPT Comb.py
│   |   └── Ordinary Kriging SPT.txt
│   └── Model comparison/
│       └── TBC
├── Others/
|   ├── 3D Rising Main - Height.py
|   ├── 3D Rising Main.py
|   ├── CrossSection - buff 5.py
|   ├── CrossSection - buff 10.py
|   ├── Lithological Profile.py
│   └── SPT Profile.py
├── Script of toolbox/
│   └── tbc
└── README.md
```
## Script Function
### 3D Modelling
#### Geological modelling
- Scripts for different modelling methods
- Create visualization model for Hong Kong and removal of land
#### Geotechnical modelling
- Script for geotechnical model using kNN
- Relationship between CPT and SPT
- Combine SPT with CPT
### Others
- Create and visualize 3D rising mains based on raw data
- Visualize and compare lithological profiles underground
- Visualize and compare SPT profiles underground
