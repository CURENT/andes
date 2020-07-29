FROM ubuntu:18.04

USER root
WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    libsuitesparse-dev \
    libopenblas-dev \
    gcc \
&& rm -rf /var/lib/apt/lists/* \
&& python -m pip install andes cvxoptklu --no-cache-dir
&& python -m pip install git+https://github.com/curent/cvxopt.git

ENTRYPOINT ["/usr/local/bin/andes"]
