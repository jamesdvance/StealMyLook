FROM public.ecr.aws/lambda/python:3.11

#RUN  apt update && apt-get install libpq-dev -y && apt-get install libpq5 -y
# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install --pre torch torchvision  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
RUN pip install -r requirements.txt

# Copy function code
COPY seg.pt ${LAMBDA_TASK_ROOT}
COPY vgg_mod.pt ${LAMBDA_TASK_ROOT}
COPY preprocessor_config.json ${LAMBDA_TASK_ROOT}
COPY api_handler.py ${LAMBDA_TASK_ROOT}
COPY handle_segment.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "api_handler.handler" ]