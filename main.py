import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from model import model
from methods import predict
from dataset import load_data, pre_process_data, output_submission


import time


TRAIN_PATH = 'data/train_cleaned.csv'
TEST_PATH = 'data/test_cleaned.csv'

train, test = load_data(TRAIN_PATH, TEST_PATH)

# get the labels values
train_raw_labels = train['SalePrice'].to_frame().as_matrix()

# pre process data sets
train_pre = pre_process_data(train)
test_pre = pre_process_data(test)

# drop unwanted columns
train_pre = train_pre.drop(['Id', 'SalePrice'], axis=1)
test_pre = test_pre.drop(['Id'], axis=1)

# align both data sets (by outer join), to make they have the same amount of features,
# this is required because of the mismatched categorical values in train and test sets
train_pre, test_pre = train_pre.align(test_pre, join='outer', axis=1)

# replace the nan values added by align for 0
train_pre.replace(to_replace=np.nan, value=0, inplace=True)
test_pre.replace(to_replace=np.nan, value=0, inplace=True)

train_pre = train_pre.as_matrix().astype(np.float)
test_pre = test_pre.as_matrix().astype(np.float)

# scale values
standard_scaler = preprocessing.StandardScaler()
train_pre = standard_scaler.fit_transform(train_pre)
test_pre = standard_scaler.fit_transform(test_pre)

X_train, X_valid, Y_train, Y_valid = train_test_split(train_pre, train_raw_labels, test_size=0.3, random_state=1)

# hyperparameters
input_size = train_pre.shape[1]
output_size = 1
num_epochs = 4000
learning_rate = 0.01
layers_dims = [input_size, 500, 500, output_size]

parameters, submission_name = model(X_train, Y_train, X_valid, Y_valid, layers_dims, num_epochs=num_epochs,
                                    learning_rate=learning_rate, print_cost=False, plot_cost=False, l2_beta=10,
                                    keep_prob=0.5, minibatch_size=0, return_best=True)
print(submission_name)

prediction = list(map(lambda val: float(val), predict(test_pre, parameters)))
output_submission(test.Id.values, prediction, 'Id', 'SalePrice', submission_name)
