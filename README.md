# 3D-Geological-Modeling
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikit-learn) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repo is for 3D geological or geotechnical modelling based on machine learning

>[!IMPORTANT]
> arcpy is not included in requirements as it needs license!

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
3D-Geological-Modeling/
├── src/                    # Source code for the project
│   ├── borehole/           # Borehole data processing modules
│   ├── modeling/           # 3D modeling algorithms (kNN, RF, SVM, etc.)
│   ├── comparison/         # Model comparison scripts
│   ├── arcpy_utils/        # Arcpy utility functions
│   └── utils/              # Shared utility functions
├── examples/               # Examples
├── requirements.txt        # Python dependencies
├── .gitignore
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
#### Model comparison
- Compare accuracy and F1 among medels
- Compare borehole and cross sections among models
### Others
- Create and visualize 3D rising mains based on raw data
- Visualize and compare lithological profiles underground
- Visualize and compare SPT profiles underground
### Script of toolbox
- Functional tools to create 3D borehole logs, models and cross sections based on GIS
