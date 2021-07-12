import argparse
import pandas as pd
import os
from sklearn import svm
from numpy import genfromtxt
from joblib import load, dump
import logging

logging.basicConfig(level=logging.INFO)
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

    # Take the set of files and read them all into a single pandas dataframe
    train_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if args.test:
        test_files = [os.path.join(args.test, file) for file in os.listdir(args.test)]
    if len(train_files) == 0 or (args.test and len(test_files)) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    X_train = genfromtxt('{}/train_x.csv'.format(args.train), delimiter=',')
    y_train = genfromtxt('{}/train_y.csv'.format(args.train), delimiter=',')

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = svm.SVC(kernel=args.kernel, C=args.c, gamma=args.gamma, verbose=1).fit(X_train, y_train)
    if args.test:
        X_test = genfromtxt('{}/test_x.csv'.format(args.test), delimiter=',')
        y_test = genfromtxt('{}/test_y.csv'.format(args.test), delimiter=',')
        logging.info('model test score:{};'.format(clf.score(X_test, y_test)))
    dump(clf, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = load(os.path.join(model_dir, "model.joblib"))
    return clf