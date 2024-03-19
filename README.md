

# CPSC542 Assignment 2

## Student Information

- **Student Name:** Devyn Miller
- **Student ID:** 2409539

## Collaboration

- **In-class collaborators:** None

## Resources

- **Resources used:**
   - [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/%7Evgg/data/pets/)
   - [oxford_iiit_pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)
   - Copilot
   - Perplexity

## Data Source

- **Data source:** [oxford_iiit_pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) `dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)`

## Code Repository

- **GitHub Repository:** [assignment2-cpsc-542](https://github.com/devyn-miller/assignment2-cpsc-542.git)

# Additional notes below — please read.

1. All of the plots and figures for this project can be found in the Jupyter notebook entitled `main.ipynb`. The report only contains the metrics table and the graphs for the loss function and accuracy.
2. I have included a folder called `not-for-submission` which contains supplementary Jupyter notebooks that were created during the development and experimentation process for the assignment, including playing around with various parameters. These notebooks are not part of the main submission but provide insights into the iterative development and exploration of different approaches. Each notebook represents a different version or variation of the main assignment, exploring different techniques, parameters, or design choices as discussed in the submitted report.

Contents:

- `CPSC542_cifar_Assignment1SeparableConv2D.ipynb`: Notebook exploring the use of separable convolutional layers.
- `CPSC542_cifar_Assignment1_grayscale.ipynb`: Notebook experimenting with grayscale conversion of images.
- `CPSC542_cifar_Assignment1batch_128.ipynb`: Notebook investigating batch size effects with a batch size of 128.
- `CPSC542_cifar_Assignment1downsample_image_res.ipynb`: Notebook exploring downsampled image resolutions.
- `CPSC542_cifar_Assignment1extra_convblock.ipynb`: Notebook with additional convolutional blocks for comparison.
- `CPSC542_cifar_Assignment1only_one_conv_block.ipynb`: Notebook focusing on using only one convolutional block.
- `CPSC542_cifar_Assignment1v2.ipynb`: Revised version of the main assignment with improvements or changes.
- `CPSC542_cifar_Assignment1v2_(2)only_two_conv_blocks_finished_running.ipynb`: Notebook with only two convolutional blocks, completed and ready for review.

These notebooks are provided for reference purposes and can be consulted to understand the development process, experiments conducted, and decisions made during the assignment. They may contain incomplete or experimental code and should not be considered as final submissions.
