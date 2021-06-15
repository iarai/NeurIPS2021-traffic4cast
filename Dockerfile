FROM continuumio/miniconda3

RUN apt update -qy &&  apt install build-essential -qy


SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

#TODO do not use root
#RUN useradd -ms /bin/bash t4c
#USER t4c
#WORKDIR /home/t4c

# add environment
ADD environment.yaml .


# https://docs.anaconda.com/anaconda/install/silent-mode/
RUN eval "$(/opt/conda/bin/conda shell.bash hook)" && \
    conda init bash && \
    source ~/.bashrc && \
    printenv && \
    export CONDA_ENVS_PATH=$PWD && \
    conda env create -f environment.yaml && \
    conda env list && \
    conda activate t4c && \
    python --version && \
    python -c 'import torch_geometric'
