FROM debian:latest

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
&& rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install andes kvxopt --no-cache-dir

RUN useradd -ms /bin/bash cui

RUN python3 -m andes selftest && \
mv /root/.andes /home/cui && \
chown -R cui:cui /home/cui/.andes

USER cui
WORKDIR /andes

ENTRYPOINT ["/usr/local/bin/andes"]
