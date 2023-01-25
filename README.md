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

# Usage 

* Running on DGX machines: 
1. Start the server: 
    ```
    python ms.py --hf_checkpoint /raid/khippe/hf_neox_radbio/gpt-neo-1.3B --tp 4
    ``` 
    Replace your HF checkpoint path with the model you would like to serve. If you have a custom model make sure the folder is in HF model format. If you would like you can just specify a HF model name and it will download it for you (`EleutherAI/gpt-neox-20b`). 

2. On your local machine, port forward the server to your local machine: 
    ```
    ssh -NL 7070:localhost:7070 [USERNAME]@rbdgx1.cels.anl.gov
    ```

3. Visit http://localhost:7070/docs to interact with the model.