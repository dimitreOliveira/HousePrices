import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split

from model import model
from methods import predict
from dataset import load_data, convert_to_one_hot, pre_process_data, output_submission


# optional, sets output of data when printed
desired_width = 320
pd.set_option('display.width', desired_width)
# np.set_printoptions(linewidth=desired_width)


TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

train, test = load_data(TRAIN_PATH, TEST_PATH)

# train = pre_process_data(train)
# test = pre_process_data(test)

# get the labels values
train_raw_labels = train['SalePrice'].to_frame().as_matrix()

# drop label
train_pre = train.drop(['SalePrice'], axis=1)

train_pre = pre_process_data(train_pre)
test_pre = pre_process_data(test)

# drop columns with null values
train_pre = train_pre.drop(['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'FireplaceQu', 'GarageType',
                            'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
                            'BsmtExposure', 'BsmtFinType1', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtFinType1',
                            'BsmtFinType2', 'Electrical', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                            'BsmtHalfBath', 'GarageCars', 'GarageArea', 'MSZoning', 'Utilities', 'Exterior1st',
                            'Exterior2nd',  'BsmtFinSF1', 'KitchenQual', 'Functional', 'SaleType'], axis=1)
test_pre = test_pre.drop(['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
                      'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                      'BsmtFinType1', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
                      'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars',
                      'GarageArea', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'KitchenQual',
                      'Functional', 'SaleType'], axis=1)

# drop columns with categorical values
train_pre = train_pre.drop(['HouseStyle', 'RoofMatl', 'Heating', 'Condition2'], axis=1)
test_pre = test_pre.drop(['HouseStyle', 'RoofMatl', 'Heating', 'Condition2'], axis=1)
# train_pre = train_pre.drop(['HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'BldgType', 'ExterCond', 'Foundation',
#                             'Heating', 'HeatingQC', 'CentralAir', 'Condition1', 'Condition2', 'Neighborhood',
#                             'LandSlope', 'LotConfig', 'LandContour', 'LotShape', 'Street', 'PavedDrive',
#                             'SaleCondition'], axis=1)
# test_pre = test_pre.drop(['HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'BldgType', 'ExterCond', 'Foundation',
#                           'Heating', 'HeatingQC', 'CentralAir', 'Condition1', 'Condition2', 'Neighborhood', 'LandSlope',
#                           'LotConfig', 'LandContour', 'LotShape', 'Street', 'PavedDrive', 'SaleCondition'], axis=1)

# drop unwanted columns
train_pre = train_pre.drop(['Id'], axis=1).as_matrix()
test_pre = test_pre.drop(['Id'], axis=1).as_matrix()

X_train, X_valid, Y_train, Y_valid = train_test_split(train_pre, train_raw_labels, test_size=0.3, random_state=1)

# hyperparameters
input_size = train_pre.shape[1]
output_size = 1
num_epochs = 10000
learning_rate = 0.01
layers_dims = [input_size, 100, output_size]

trained_parameters, submission_name = model(X_train, Y_train, X_valid, Y_valid, layers_dims, num_epochs=num_epochs,
                                            learning_rate=learning_rate, use_l2=False, use_dropout=False,
                                            print_cost=False, plot_cost=True, l2_beta=0.01, keep_prob=0.8,
                                            return_max_acc=False)
print(submission_name)

prediction = list(map(lambda val: float(val), predict(test_pre, trained_parameters)))
output_submission(test.Id.values, prediction, 'Id', 'SalePrice', submission_name)
