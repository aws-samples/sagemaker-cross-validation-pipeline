import argparse
import pandas as pd
import os
from sklearn import svm
from numpy import genfromtxt
from joblib import load, dump
import logging
import json

logging.basicConfig(level=logging.INFO)

base_dir = "/opt/ml/processing"
base_dir_evaluation = f"{base_dir}/evaluation" 

def train(train=None, test=None):
    """Trains a model using the specified algorithm with given parameters.
    
       Args:
          train : location on the filesystem for training dataset
          test: location on the filesystem for test dataset 
          
       Returns:
          trained model object
    """
    # Take the set of files and read them all into a single pandas dataframe
    train_files = [ os.path.join(train, file) for file in os.listdir(train) ]
    if test:
        test_files = [os.path.join(test, file) for file in os.listdir(test)]
    if len(train_files) == 0 or (test and len(test_files)) == 0:
        raise ValueError((f'There are no files in {train}.\n' +
                          'This usually indicates that the channel train was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.'))

    X_train = genfromtxt(f'{train}/train_x.csv', delimiter=',')
    y_train = genfromtxt(f'{train}/train_y.csv', delimiter=',')

    # Now use scikit-learn's decision tree classifier to train the model.
    if "SM_CHANNEL_JOBINFO" in os.environ: 
      jobinfo_path = os.environ.get('SM_CHANNEL_JOBINFO')
      with open(f"{jobinfo_path}/jobinfo.json", "r") as f:
        jobinfo = json.load(f)
        hyperparams = jobinfo['hyperparams']
        clf = svm.SVC(kernel=hyperparams['kernel'], 
                      C=float(hyperparams['c']), 
                      gamma=float(hyperparams['gamma']), 
                      verbose=1).fit(X_train, y_train)
    else:     
      clf = svm.SVC(kernel=args.kernel, 
                    C=args.c, 
                    gamma=args.gamma, 
                    verbose=1).fit(X_train, y_train)
    return clf
    
def evaluate(test=None, model=None):
    """Evaluates the performance for the given model.
    
       Args:
          test: location on the filesystem for test dataset 
    """
    if test:
        X_test = genfromtxt(f'{test}/test_x.csv', delimiter=',')
        y_test = genfromtxt(f'{test}/test_y.csv', delimiter=',')
        accuracy_score = model.score(X_test, y_test)
        logging.info(f'model test score:{accuracy_score};')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('-c', '--c', type=float, default=1.0)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--kernel', type=str)
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()
    model = train(train=args.train, test=args.test)

    evaluate(test=args.test, model=model)
    dump(model, os.path.join(args.model_dir, "model.joblib"))
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = load(os.path.join(model_dir, "model.joblib"))
    return clf
