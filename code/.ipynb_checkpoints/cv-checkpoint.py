#!/usr/bin/env python
import os
import boto3
import re
import json
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import argparse
import time
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO)

sklearn_framework_version='0.23-1'
script_path = 'scikit_learn_iris.py'

region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()

sm_client = boto3.client("sagemaker")

def fit_model(instance_type, 
              output_path,
              s3_train_base_dir,
              s3_test_base_dir,
              f, 
              c, 
              gamma, 
              kernel
             ):
    """Fits a model using the specified algorithm.
    
       Args:
            instance_type: instance to use for Sagemaker Training job
            output_path: S3 URI as the location for the trained model artifact
            s3_train_base_dir: S3 URI for train datasets
            s3_test_base_dir: S3 URI for test datasets
            f: index represents a fold number in the K fold cross validation
            c: regularization parameter for SVM
            gamma: kernel coefficiency value
            kernel: kernel type for SVM algorithm
       
       Returns: 
            Sagemaker Estimator created with given input parameters.
    """
    sklearn_estimator = SKLearn(
        entry_point=script_path,
        instance_type=instance_type,
        framework_version=sklearn_framework_version,
        role=role,
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        hyperparameters={'c': c, 'gamma' : gamma, 'kernel': kernel},
        metric_definitions= [ { "Name": "test:score", "Regex": "model test score:(.*?);" }]
    )
    sklearn_estimator.fit(inputs = { 'train': f'{s3_train_base_dir}/{f}',
                                     'test':  f'{s3_test_base_dir}/{f}'
                                   }, wait=False)
    return sklearn_estimator

def monitor_training_jobs(training_jobs):
    """Monitors the submit training jobs for completion.
    
      Args: 
         training_jobs: array of submitted training jobs
         
    """
    all_jobs_done = False
    while not all_jobs_done:
        completed_jobs = 0
        for job in training_jobs:
           job_detail = sm_client.describe_training_job(TrainingJobName=job._current_job_name)
           job_status = job_detail['TrainingJobStatus']
           if job_status.lower() in ('completed', 'failed', 'stopped'):
               completed_jobs += 1
        if completed_jobs == len(training_jobs):
            all_jobs_done = True
        else:
            time.sleep(30)

def evaluation(training_jobs):
    """Evaluates and calculate the performance for the cross validation training jobs.
    
       Args:
         training_jobs: array of submitted training jobs
      
       Returns:
         Average score from the training jobs collection in the given input
    """
    scores = []
    for job in training_jobs:
        job_detail = sm_client.describe_training_job(TrainingJobName=job._current_job_name)
        metrics = job_detail['FinalMetricDataList']
        score = [x['Value'] for x in metrics if x['MetricName'] == 'test:score'][0]
        scores.append(score)
        
    np_scores = np.array(scores)
    
    # Calculate the score by taking the average score across the performance of the training job
    score_avg = np.average(np_scores)
    logging.info(f'average model test score:{score_avg};')
    return score_avg

def train():
    """
    Trains a Cross Validation Model with the given parameters.
    
    """
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('-c', type=float, default=1.0)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--kernel', type=str)
    parser.add_argument('-k', '--k', type=int, default=5)
    parser.add_argument('--train_src', type=str)
    parser.add_argument('--test_src', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--instance_type', type=str, default="ml.c4.xlarge")
    
    args = parser.parse_args()

    training_jobs = []
    # Fit k training jobs with the specified parameters.
    for f in range(args.k):
        sklearn_estimator = fit_model(instance_type=args.instance_type,
                                      output_path=args.output_path,
                                      s3_train_base_dir=args.train_src,
                                      s3_test_base_dir=args.test_src,
                                      f=f,
                                      c=args.c,
                                      gamma=args.gamma,
                                      kernel=args.kernel)
        training_jobs.append(sklearn_estimator)
        time.sleep(5) # sleeps to avoid Sagemaker Training Job API throttling

    monitor_training_jobs(training_jobs=training_jobs)
    score = evaluation(training_jobs=training_jobs)
    return score

if __name__ == '__main__':
    train()
    sys.exit(0)

