from sagemaker.estimator import Estimator
import sagemaker
import argparse
import os
import json
import boto3
from sagemaker.tuner import ParameterRange, CategoricalParameter, ContinuousParameter, HyperparameterTuner

sagemaker_session = sagemaker.session.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()
sm_client = boto3.client("sagemaker")

base_dir = "/opt/ml/processing"
base_dir_evaluation = f"{base_dir}/evaluation" 
base_dir_jobinfo = f"{base_dir}/jobinfo" 

def train(train=None, 
          test=None, 
          image_uri=None, 
          instance_type="ml.c4.xlarge", 
          instance_count=1, 
          output_path=None,
          k = 5,
          max_jobs=2,
          max_parallel_jobs=2,
          min_c = 0,
          max_c = 1,
          min_gamma=0.0001,
          max_gamma=0.001,
          gamma_scaling_type="Logarithmic"):
    
    """Triggers a sagemaker automatic hyperparameter tuning optimization job to train and evaluate a given algorithm. 
     Hyperparameter tuner job triggers maximum number of training jobs with the given maximum parallel jobs per batch. Each training job triggered by the tuner would trigger k cross validation model training jobs.
     
     Args:
         train: S3 URI where the training dataset is located
         test: S3 URI where the test dataset is located
         image_uri: ECR repository URI for the training image
         instance_type: Instance type to be used for the Sagemaker Training Jobs.
         instance_count: number of intances to be used for the Sagemaker Training Jobs.
         output_path: S3 URI for the output artifacts generated in this script.
         k: number of k in Kfold cross validation
         max_jobs: Maximum number of jobs the HyperparameterTuner triggers
         max_parallel_jobs: Maximum number of parallel jobs the HyperparameterTuner trigger in one batch.
         min_c: minimum c value configure as continuous parameter for hyperparameter tuning process
         max_c: maximum c value configure as continuous parameter for hyperparameter tuning process
         min_gamma: minimum gamma value configure as continuous parameter for hyperparameter tuning process
         max_gamma: maximum gamma value configure as continuous parameter for hyperparameter tuning process
         gamma_scaling_type: scaling type used in the Hyperparameter tuning process for gamma
    """

  # An Estimator object to be associated with the HyperparameterTuner job. 
    cv_estimator = Estimator(
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=instance_count,
        role=role,
        sagemaker_session=sagemaker_session,
        output_path=output_path)

    cv_estimator.set_hyperparameters(
        train_src=train,
        test_src = test,
        k = k,
        instance_type=instance_type)

    hyperparameter_ranges = {
                            'c': ContinuousParameter(min_c, max_c), 
                            'kernel' : CategoricalParameter(['linear', 'poly', 'rbf', 'sigmoid']),
                            'gamma' : ContinuousParameter(min_value=min_gamma, 
                                                          max_value=max_gamma, 
                                                          scaling_type=gamma_scaling_type)
                          }

    objective_metric_name = "test:score"
    tuner = HyperparameterTuner(cv_estimator,
                              objective_metric_name,
                              hyperparameter_ranges,
                              objective_type="Maximize",
                              max_jobs=max_jobs,
                              max_parallel_jobs=max_parallel_jobs,
                              metric_definitions=[{"Name": objective_metric_name, 
                                                   "Regex": "model test score:(.*?);"}])

    tuner.fit({"train": train, "test": test}, include_cls_metadata=True)

    best_traning_job_name = tuner.best_training_job()
    tuner_job_name = tuner.latest_tuning_job.name  
    best_performing_job = sm_client.describe_training_job(TrainingJobName=best_traning_job_name)

    hyper_params = best_performing_job['HyperParameters']
    best_hyperparams = { k:v for k,v in hyper_params.items() if not k.startswith("sagemaker_")}

    jobinfo = {}
    jobinfo['name'] = tuner_job_name
    jobinfo['best_training_job'] = best_traning_job_name
    jobinfo['hyperparams'] = best_hyperparams
    metric_value = [ x['Value'] for x in best_performing_job['FinalMetricDataList'] 
                    if x['MetricName'] == objective_metric_name ][0]
    
    evaluation_metrics = { "multiclass_classification_metrics" : {
                            "accuracy" : {
                              "value" : metric_value,
                              "standard_deviation" : "NaN"
                            },
                         }
                       }
    os.makedirs(base_dir_evaluation, exist_ok=True) 
    with open(f'{base_dir_evaluation}/evaluation.json', 'w') as f:
        f.write(json.dumps(evaluation_metrics))

    with open(f'{base_dir_jobinfo}/jobinfo.json', 'w') as f:
        f.write(json.dumps(jobinfo))
  
if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('-k', '--k', type=int, default=5)
    parser.add_argument('--image-uri', type=str)    
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--instance-type', type=str, default="ml.c4.xlarge")
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--max-jobs', type=int, default=2)
    parser.add_argument('--max-parallel-jobs', type=int, default=2)
    parser.add_argument('--min-c', type=int, default=0)
    parser.add_argument('--max-c', type=int, default=1)
    parser.add_argument('--min-gamma', type=float, default=0.0001)
    parser.add_argument('--max-gamma', type=float, default=0.001)
    parser.add_argument('--gamma-scaling-type', type=str, default="Logarithmic")
    
    args = parser.parse_args()
    train(train=args.train, 
          test=args.test, 
          image_uri=args.image_uri, 
          instance_type=args.instance_type, 
          instance_count=args.instance_count,
          output_path=args.output_path,
          k=args.k,
          max_jobs=args.max_jobs,
          max_parallel_jobs=args.max_parallel_jobs,
          min_c=args.min_c,
          max_c=args.max_c,
          min_gamma=args.min_gamma,
          max_gamma=args.max_gamma,
          gamma_scaling_type=args.gamma_scaling_type)