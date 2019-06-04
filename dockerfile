FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.5.8

copy . /featuretools_ta1

RUN pip install -e featuretools_ta1
