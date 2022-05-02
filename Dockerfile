FROM nvcr.io/nvidia/deepstream-l4t:6.0.1-triton

RUN apt-get update && apt-get -y upgrade

WORKDIR /opt/nvidia/deepstream/deepstream-6.0

EXPOSE 8554

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
CMD ["/bin/bash"]
#EOF
