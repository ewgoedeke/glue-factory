# 3D-Vision Project ReadME

Its apparent that we implemented our project using the glue-factory framework of the CVG group. This file explains
where our code is located and how you can run the things we ran.

## Contribution

We implemeneted our model as well as some training infrastructure and helper code.

### gluefactory/models/extractors/jpldd

is the package containing the implementation of our model. While the file "joint_point_line_extractor.py" is the main file of our model and contains the full pipeline. The other files contain submodules or helpers.

### gluefactory/configs

Contains configurations of training and evaluation runs. Our own configurations start with "jpldd".

### gluefactory/ground_truth_generation

contains the scripts we used to generate the groundtruth for points and lines. it also contains "timing.py" which was used to conduct timing measurements.

### notebooks/

Contains jupyter notebooks that we used to test our method and visualize results.

### cluster/

Contains scripts and the settings file we used to run training and evaluations on the cluster.

### gluefactory/datasets

We implemented our own dataset in `minidepth.py`.

### gluefactory/eval

We implemented a custom evaluation pipeline for our line detection. it can be found in `gluefactory/eval/hpatches_lines.py`.

## Test our method

If you want to test our implementation you can use the notebook `notebooks/JPLDD_demo_notebook.ipynb`. There you can find an easy to follow walk-through for the functionality of JPLDD on a single image. It includes a demonstration of:

- Heatmap generation
- Keypoint detection
- Line Detection
- Descriptor extraction

Make sure to install the dependencies according to the original README.md in this repo.
