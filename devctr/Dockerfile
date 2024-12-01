FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
LABEL authors="Jordan Lam jordlam@mit.edu, Keawe Mann keawem@mit.edu"

ARG DEBIAN_FRONTEND=noninteractive

# Install needed packages.
RUN apt-get update && apt update
RUN apt -y install python3

# Huh? https://github.com/jenkinsci/docker/issues/543
RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy true;" >> /etc/apt/apt.conf.d/99custom
RUN apt -y install git

RUN git clone https://github.com/tinyobjloader/tinyobjloader.git
RUN git clone https://github.com/nothings/stb.git
RUN git clone https://github.com/ufbx/ufbx.git
