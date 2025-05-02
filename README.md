# Adaptive Learning from Multi-Source Motion Sensor Data
This repository contains an implementation of the paper found here: https://arxiv.org/pdf/2405.16395

# Run via Google Colab
The easiest way to run the code is by using a T4 GPU on Google Colab. The CS_598_DLH_final.ipynb notebook can be run directly within Colab. 

# Run via bare metal
For the hardcode, you can run this model on a computer with a GPU. To get started, run
`python -m venv venv`
`source venv/bin/activate`
`pip install -r bare_metal/requirements.txt` 

*Note that this does not install any NVIDIA drivers, because that is unique based on what OS/execution environment/GPU is being used. That is left for the reader to figure out. Good luck!*