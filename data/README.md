# Dataset description and details

Currently only victoria park dataset is included.

## Structure
```text
data/
├── victoria_park/
│   ├── raw/
│   └── processed/
└──  README.md
```

### victoria_park/
raw/ contains the original dataset files as downloaded from the source https://www-personal.acfr.usyd.edu.au/nebot/victoria_park.htm. This is the data that is used in the SLAM pipeline. See info.txt in that folder for details.

processed/ contains preprocessed versions of the victoria park data. Not used in the pipeline, but may be convenient for quick experiments or visualization. 
- victoria_park.txt is retrieved from the GTSAM data-folder https://github.com/borglab/gtsam/blob/develop/examples/Data/victoria_park.txt. Here odometry is given as relative poses and landmark detections are given as range-bearing measurements with known correspondences. Could not find info on exactly how this is processed. 
- victoriaParkDataset.mat is retrived from Matlab webiste ""

Could not find 

