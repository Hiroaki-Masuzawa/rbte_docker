FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y git libopencv-dev
RUN pip install pyyaml Pillow numpy opencv-contrib-python scikit-image

RUN apt install -y imagemagick
RUN pip install pandas