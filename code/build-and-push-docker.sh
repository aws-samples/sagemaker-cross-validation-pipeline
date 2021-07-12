#!/bin/bash

if [ $# -lt 2 ]
then
  echo "usage: build-and-push-docker.sh <aws_acct_id> <aws_region>"
  exit 1
fi

aws_account_id=$1
region=$2
export AWS_DEFAULT_REGION=${region}

repo_name="sagemaker-cross-validation-pipeline"
version="latest"
image_name="${repo_name}:${version}"

docker build -t ${image_name} .
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${aws_account_id}.dkr.ecr.${region}.amazonaws.com
aws ecr describe-repositories --repository-names ${repo_name}
repo_exist=$?

if [ ${repo_exist} -ne 0 ]
then
 echo "repository doesn't exist. Create one now"
 aws ecr create-repository \
    --repository-name  ${repo_name} \
    --image-scanning-configuration scanOnPush=true \
    --region ${region}
fi
set -e
docker tag ${image_name} ${aws_account_id}.dkr.ecr.${region}.amazonaws.com/${image_name}
docker push ${aws_account_id}.dkr.ecr.${region}.amazonaws.com/${image_name}
echo "Docker Image ${aws_account_id}.dkr.ecr.${region}.amazonaws.com/${image_name} has been push to ECR successfully"
