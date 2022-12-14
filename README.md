# Serving models for RadBio 

# Installation 

1. Install `pytorch` (I am using v1.13.0, CUDA 11.6):

    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    ```
1. Install requirements and package 

    ```
    pip3 install -r requirements/requirements.txt
    pip3 install . 
    ```