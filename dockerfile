FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18-20200630-050709

# RUN pip install --upgrade pip

RUN pip uninstall -y featuretools

copy . /featuretools_ta1

RUN pip install -e featuretools_ta1

ADD runtime.py /usr/local/lib/python3.6/dist-packages/d3m/runtime.py\
