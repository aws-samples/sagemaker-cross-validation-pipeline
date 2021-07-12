# Build a Highly Scalable Cross Validation Training Pipeline With Sagemaker
This repository contains source code and a jupyter notebook that triggers the cross validation training pipeline using Sagemaker Pipeline. 


## Goals
This main goal of this project is to provide a reference implementation for building a cross validation training pipeline with Amazon Sagemaker. By running the supplement juypter notebook provided in this project, you would have built a cross validation training pipeline that integrates with [Sagemaker Pipelines](https://aws.amazon.com/sagemaker/pipelines/) and [Sagemaker Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html). Typically, training cross validation models involve k fold models trained in sequence, usually performed on the same server. In this project, we implements a technique that leverages Sagemaker SDK, Sagemaker Hyperparameter Tuner and Sagemaker Training Jobs to train k fold models in parallel to improve scalability and performance.

This project also leverages custom docker container, therefore, by running this example you would have an idea how to go about running a Sagemaker training job using a custom container.

## Architecture
![architecture diagram](assets/crossvalidationpipeline.png)

## License
This library is licensed under the MIT-0 License. See the LICENSE file.

<details>
<summary>  
<b>External Dependencies</b>

This package depends on and may retrieve a number of third-party software packages (such as open source packages) from third-party servers at install-time or build-time ("External Dependencies"). The External Dependencies are subject to license terms that you must accept in order to use this package. If you do not accept all of the applicable license terms, you should not use this package. We recommend that you consult your company’s open source approval policy before proceeding.

</summary>
Provided below is a list of the External Dependencies and the applicable license terms as indicated by the documentation associated with the External Dependencies as of Amazon's most recent review of such documentation.
THIS INFORMATION IS PROVIDED FOR CONVENIENCE ONLY. AMAZON DOES NOT PROMISE THAT THE LIST OR THE APPLICABLE TERMS AND CONDITIONS ARE COMPLETE, ACCURATE, OR UP-TO-DATE, AND AMAZON WILL HAVE NO LIABILITY FOR ANY INACCURACIES. YOU SHOULD CONSULT THE DOWNLOAD SITES FOR THE EXTERNAL DEPENDENCIES FOR THE MOST COMPLETE AND UP-TO-DATE LICENSING INFORMATION.
YOUR USE OF THE EXTERNAL DEPENDENCIES IS AT YOUR SOLE RISK. IN NO EVENT WILL AMAZON BE LIABLE FOR ANY DAMAGES, INCLUDING WITHOUT LIMITATION ANY DIRECT, INDIRECT, CONSEQUENTIAL, SPECIAL, INCIDENTAL, OR PUNITIVE DAMAGES (INCLUDING FOR ANY LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, OR COMPUTER FAILURE OR MALFUNCTION) ARISING FROM OR RELATING TO THE EXTERNAL DEPENDENCIES, HOWEVER CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY, EVEN IF AMAZON HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. THESE LIMITATIONS AND DISCLAIMERS APPLY EXCEPT TO THE EXTENT PROHIBITED BY APPLICABLE LAW.

** sklearn; version 0.22.1 -- https://scikit-learn.org
</details>

## Step 1 - Build A Docker Image
Before executing any of the cells in the given jupyter notebook, we need to build a docker image using the shell script provided in the project (in the code folder).
```
./build-and-push-docker.sh [aws_acct_id] [aws_region]
```

Capture the ECR repository name from the script after a successful run. You'll need to provide the image name at pipeiline execution time. Here's an example of a valid ECR repo name: 869530972998.dkr.ecr.us-east-2.amazonaws.com/sagemaker-cross-validation-pipeline:latest

## Step 2 - Update Pipeline Parameters
Following items are a list of variables used in pipeline definition. These values can be overwritten at pipeline execution time for different results. 

- **processing_instance_count** - number of instances for a Sagemaker Processing job in prepropcessing step.
- **processing_instance_type** - instance type used for a Sagemaker Processing job in prepropcessing step.
- **training_instance_type** - instance type used for Sagemaker Training job.
- **training_instance_count** - number of instances for a Sagemaker Training job.
- **inference_instance_type** - instance type for hosting the deployment of the Sagemaker trained model.
- **hpo_tuner_instance_type** - instance type for the script processor that triggers the hyperparameter tuning job
- **model_approval_status** - the initial approval status for the trained model in Sagemaker Model Registry
- **role** - IAM role to use throughout the specific pipeline execution.
- **default_bucket** - default S3 bucket name as the object storage for the target pipeline execution.
- **baseline_model_objective_value** - the minimum objective metrics used for model evaluation.
- **bucket_prefix** - bucket prefix for the pipeline execution.
- **image_uri** = docker image URI (ECR) for triggering cross validation model training with HyperparameterTuner.
- **k** - the value of k to be used in k fold cross validation
- **max_jobs** - maximum number of model training jobs to trigger in a single hyperparameter tuner job.
- **max_parallel_jobs** - maximum number of parallel model training jobs to trigger in a single hyperparameter tuner job.

To update any variables, open the [jupyter notebook](cross_validation_pipeline.ipynb), navigate towards the bottom of the notebook where the pipeline execution is triggered and update the parameters  with the desired values:

```
execution = pipeline.start(
    parameters=dict(
        BaselineModelObjectiveValue=0.8,
        MinimumC=0,
        MaximumC=1,
        image_uri="869530972998.dkr.ecr.us-east-2.amazonaws.com/sagemaker-cross-validation-pipeline:latest"
    ))
```

**Note** Replace the *image_uri* with the docker image URI that you built and pushed described in Step 1

## Step 3 - Trigger Pipeline Run
Execute all the cells in the notebook from the beginning. The notebook should trigger pipeline execution at the end. If you use Sagemaker Studio, you can visualize the pipeline execution by navigate to Sagemaker Pipelines in the left hand panel. 

![sagemaker Studio Pipeline](assets/sagemaker-studio-pipeline.png)

You can track the status of pipeline execution directly from the pipeline dashboard:

![sagemaker Studio Pipeline Dashboard](assets/sagemaker-studio-pipeline-dashboard.png)

To drill down to specific job, double click the execution from the dashboard:

[sagemaker Studio Pipeline Execution](assets/sagemaker-studio-pipeline-execution.png)


