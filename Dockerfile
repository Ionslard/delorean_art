# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim or tensorflow base

#      Standard version
FROM python:3.10

#      Slim version
# FROM python:3.10-slim

#      Tensorflow version (attention: won't run on Apple Silicon)
# FROM tensorflow/tensorflow:2.16.1

# Copy everything we need into the image
#COPY packagename packagename             # attention à updater
COPY api api
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY models models
# COPY credentials.json credentials.json

# LIBGL pour lire openCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Install everything
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .
# RUN pip install python-multipart

# Make directories that we need, but that are not included in the COPY
RUN mkdir /raw_data
# RUN mkdir /models

# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
