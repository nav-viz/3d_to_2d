# Description

- The MFCAD++ dataset is a machining feature recognition dataset containing B-Rep CAD models.
- Each CAD model has been saved as a STEP file.
- The CAD models were automatically generated using the PythonOCC CAD software.
- For each CAD model, a machining feature class label is given on each B-Rep face.
- These labels can easily be extracted from the STEP files.
- The labels are given in "feature_labels.txt".
- The dataset has been split into "train", "val" and "test" directories using a 70:15:15 split as per the original paper.
- These splits are given in the files: "train.txt", "val.txt" and "test.txt".
- There are training_set=41766, val_set=8950 & test_set=8949 samples, with 59665 samples in total.
- For more information on hierachical B-Rep graphs store in H5DF files see "h5_structure.h5".

If used please reference the paper: 
Colligan AR, Robinson TR, Nolan DC, Hua Y, Cao W. Hierarchical CADNet: Learning from B-Reps for Machining Feature Recognition, Computer-Aided Design

