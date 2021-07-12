FROM python:3.7
RUN apt-get update && pip install sagemaker boto3 numpy sagemaker-training
COPY cv.py /opt/ml/code/train.py
COPY scikit_learn_iris.py /opt/ml/code/scikit_learn_iris.py
ENV SAGEMAKER_PROGRAM train.py