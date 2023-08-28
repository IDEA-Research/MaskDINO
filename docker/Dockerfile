ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"
# ARG python=3.7
# ENV PYTHON_VERSION=${python}

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get -y update
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git
RUN conda clean --all
RUN git clone https://github.com/phanthanhhang/MaskDINO.git /MaskDINO
WORKDIR /MaskDINO
RUN chmod 777 /MaskDINO
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# RUN git submodule init
# RUN git submodule update
# RUN python -m pip install -e detectron2
# RUN pip install -e .