

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
ENV NVIDIA_DRIVER_CAPABILITIES all

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        software-properties-common curl vim git zip unzip unrar p7zip-full wget cmake \
        apache2 openssl libssl-dev

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libwebp-dev libcairo2-dev libjpeg-dev libgif-dev \
        libboost-all-dev libopencv-dev libwxgtk3.0-gtk3-dev \
        ffmpeg libgl1-mesa-glx libsm6 libxext6 libxrender-dev libx11-xcb1 \
        mesa-utils xauth xorg openbox xvfb

RUN conda install \
    'pillow==8.4.0' \
    'matplotlib==3.3.4' \
    'pandas==1.2.4' \
    'scipy==1.6.2' \
    'scikit-image==0.18.1' \
    'jupyterlab==3.0.14' \
&& conda install -c conda-forge \
    'pytorch-lightning==1.3.8' \
    'scikit-learn==0.23.2'

RUN pip install \
    'mysql-connector-python==8.0.25' \
    'patool==1.12' \
    'pyunpack==0.2.2' \
    'opencv-python==4.5.2.52' \
    'opencv-contrib-python==4.5.3.56' \
    'kornia==0.5.2' \
    'cupy-cuda111==9.4.0' \
    'imagesize==1.2.0' \
    'lpips==0.1.4' \
    'torchmetrics==0.6.0' \
&& pip install \
    --extra-index-url https://developer.download.nvidia.com/compute/redist \
    --upgrade 'nvidia-dali-cuda110==1.3.0'





