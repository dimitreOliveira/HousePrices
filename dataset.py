import csv
import pandas as pd


def load_data(train_path, test_path):
    """
    method for data loading
    :param train_path: path for the train set file
    :param test_path: path for the test set file
    :return: a 'pandas' array for each set
    """

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print("number of training examples = " + str(train_data.shape[0]))
    print("number of test examples = " + str(test_data.shape[0]))
    print("train shape: " + str(train_data.shape))
    print("test shape: " + str(test_data.shape))

    return train_data, test_data


def output_submission(test_ids, predictions, id_column, predction_column, file_name):
    """
    :param test_ids: vector with test dataset ids
    :param predictions: vector with test dataset predictions
    :param id_column: name of the output id column
    :param predction_column: name of the output predction column
    :param file_name: string for the output file name
    :return: output a csv with ids ands predictions
    """

    print('Outputting submission...')
    with open('submissions/' + file_name, 'w') as submission:
        writer = csv.writer(submission)
        writer.writerow([id_column, predction_column])
        for test_id, test_prediction in zip(test_ids, predictions):
            writer.writerow([test_id, test_prediction])
    print('Output complete')


def replace_na_with_mode(dataset, column_name):
    """
    :param dataset: data set
    :param column_name: column to perform function
    :return: updated data set
    """
    dataset.loc[dataset[column_name].isnull(), column_name] = dataset[column_name].mode()[0]


def replace_na_with_median(dataset, column_name):
    """
    :param dataset: data set
    :param column_name: column to perform function
    :return: updated data set
    """
    dataset.loc[dataset[column_name].isnull(), column_name] = dataset[column_name].median()


def pre_process_data(df):
    """
    Perform a number of pre process functions on the data set
    :param df: pandas data frame
    :return: processed data frame
    """

    # one-hot encode categorical values
    df = pd.get_dummies(df, columns=['ExterQual'])
    df = pd.get_dummies(df, columns=['ExterCond'])
    df = pd.get_dummies(df, columns=['Foundation'])
    df = pd.get_dummies(df, columns=['HeatingQC'])
    df = pd.get_dummies(df, columns=['Condition1'])
    df = pd.get_dummies(df, columns=['Neighborhood'])
    df = pd.get_dummies(df, columns=['SaleCondition'])
    df = pd.get_dummies(df, columns=['HouseStyle'])
    df = pd.get_dummies(df, columns=['RoofMatl'])
    df = pd.get_dummies(df, columns=['Heating'])
    df = pd.get_dummies(df, columns=['Condition2'])
    df = pd.get_dummies(df, columns=['GarageQual'])
    df = pd.get_dummies(df, columns=['Electrical'])
    df = pd.get_dummies(df, columns=['Utilities'])
    df = pd.get_dummies(df, columns=['Exterior1st'])
    df = pd.get_dummies(df, columns=['Exterior2nd'])
    df = pd.get_dummies(df, columns=['MasVnrType'])
    df = pd.get_dummies(df, columns=['GarageType'])
    df = pd.get_dummies(df, columns=['GarageFinish'])
    df = pd.get_dummies(df, columns=['GarageCond'])
    df = pd.get_dummies(df, columns=['BsmtQual'])
    df = pd.get_dummies(df, columns=['BsmtCond'])
    df = pd.get_dummies(df, columns=['BsmtExposure'])
    df = pd.get_dummies(df, columns=['BsmtFinType1'])
    df = pd.get_dummies(df, columns=['BsmtFinType2'])
    df = pd.get_dummies(df, columns=['MSZoning'])
    df = pd.get_dummies(df, columns=['KitchenQual'])
    df = pd.get_dummies(df, columns=['Functional'])
    df = pd.get_dummies(df, columns=['SaleType'])

    df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1}).astype(int)
    df['Street'] = df['Street'].map({'Grvl': 0, 'Pave': 1}).astype(int)
    df['PavedDrive'] = df['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2}).astype(int)

    return df


def mini_batches(train_set, train_labels, mini_batch_size):
    """
    Generate mini batches from the data set (data and labels)
    :param train_set: data set with the examples
    :param train_labels: data set with the labels
    :param mini_batch_size: mini batch size
    :return: mini batches
    """
    set_size = train_set.shape[0]
    batches = []
    num_complete_minibatches = set_size // mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_x = train_set[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_y = train_labels[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if set_size % mini_batch_size != 0:
        mini_batch_x = train_set[(set_size - (set_size % mini_batch_size)):]
        mini_batch_y = train_labels[(set_size - (set_size % mini_batch_size)):]
        mini_batch = (mini_batch_x, mini_batch_y)
        batches.append(mini_batch)

    return batches
