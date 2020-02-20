FROM tensorflow/tensorflow

MAINTAINER Breck Yunits <byunits@cc.hawaii.edu> 

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/lanagarmire/deepimpute && cd deepimpute && pip install --user .