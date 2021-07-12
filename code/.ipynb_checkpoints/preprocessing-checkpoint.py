import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import KFold
import os
import argparse

# Defines the base target directory for datasets.
base_dir = '/opt/ml/processing'

def save_all_datasets(X, y):
    """ Saves the entire dataset as input for model selection process in 
    the cross validation training pipeline.

    Args:
        X : numpy array represents the features
        y : numpy array represetns the target
    """
    os.makedirs(f'{base_dir}/train/all', exist_ok=True)
    np.savetxt(f'{base_dir}/train/all/train_x.csv', X, delimiter=',')
    np.savetxt(f'{base_dir}/train/all/train_y.csv', y, delimiter=',')

def save_kfold_datasets(X, y, k):
    """ Splits the datasets (X,y) k folds and saves the output from 
    each fold into separate directories.

    Args:
        X : numpy array represents the features
        y : numpy array represetns the target
        k : int value represents the number of folds to split the given datasets
    """

    # Shuffles and Split dataset into k folds
    kf = KFold(n_splits=k, random_state=None, shuffle=True)

    fold_idx = 0
    for train_index, test_index in kf.split(X, y=y, groups=None):    
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       os.makedirs(f'{base_dir}/train/{fold_idx}', exist_ok=True)
       np.savetxt(f'{base_dir}/train/{fold_idx}/train_x.csv', X_train, delimiter=',')
       np.savetxt(f'{base_dir}/train/{fold_idx}/train_y.csv', y_train, delimiter=',')

       os.makedirs(f'{base_dir}/test/{fold_idx}', exist_ok=True)
       np.savetxt(f'{base_dir}/test/{fold_idx}/test_x.csv', X_test, delimiter=',')
       np.savetxt(f'{base_dir}/test/{fold_idx}/test_y.csv', y_test, delimiter=',')
       fold_idx += 1
    
def process(k):
    """Performs preprocessing by splitting the datasets into k folds.
    """
    # Downloads Iris dataset from sklearn
    X, y = datasets.load_iris(return_X_y=True)
    save_all_datasets(X,y)
    save_kfold_datasets(X,y,k)
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k', type=int, default=5)    
    args = parser.parse_args()
    process(k=args.k)
    
