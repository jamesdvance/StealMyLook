#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker
#FROM anibali/pytorch:2.0.1-cuda11.8
FROM python:3.11.7-bullseye


# Copy requirements.txt
COPY ./requirements.txt ./requirements.txt

RUN  apt update && apt-get install libpq-dev -y && apt-get install libpq5 -y

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY ./predict.py ./predict.py
COPY ./handle_segment.py ./handle_segment.py
COPY ./vgg_mod.pt ./vgg_mod.pt
COPY ./seg.pt ./seg.pt
COPY ./preprocessor_config.json ./preprocessor_config.json

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "80"]