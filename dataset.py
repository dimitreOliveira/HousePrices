import math
import csv
import re
import numpy as np
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


def convert_to_one_hot(dataset_size, raw_labels, classes):
    """
    :param dataset_size: size of the data set
    :param raw_labels: array with the labels set
    :param classes: number of output classes
    :return: one hot labels array
    """
    labels = np.zeros((dataset_size, classes))
    labels[np.arange(dataset_size), raw_labels] = 1

    return labels


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
    :return: updated data frame
    """
    # convert 'Sex' values
    # df['gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # We see that 2 passengers embarked data is missing, we fill those in as the most common Embarked value
    # replace_na_with_mode(df, 'Embarked')

    # Replace missing age values with median ages by gender
    # for gender in df['gender'].unique():
    #     median_age = df[(df['gender'] == gender)].Age.median()
    #     df.loc[(df['Age'].isnull()) & (df['gender'] == gender), 'Age'] = median_age

    # one-hot encode categorical values
    # df = pd.get_dummies(df, columns=['HouseStyle'])
    df = pd.get_dummies(df, columns=['RoofStyle'])
    # df = pd.get_dummies(df, columns=['RoofMatl'])
    df = pd.get_dummies(df, columns=['ExterQual'])
    df = pd.get_dummies(df, columns=['BldgType'])
    df = pd.get_dummies(df, columns=['ExterCond'])
    df = pd.get_dummies(df, columns=['Foundation'])
    # df = pd.get_dummies(df, columns=['Heating'])
    df = pd.get_dummies(df, columns=['HeatingQC'])
    df = pd.get_dummies(df, columns=['CentralAir'])
    df = pd.get_dummies(df, columns=['Condition1'])
    # df = pd.get_dummies(df, columns=['Condition2'])
    df = pd.get_dummies(df, columns=['Neighborhood'])
    df = pd.get_dummies(df, columns=['LandSlope'])
    df = pd.get_dummies(df, columns=['LotConfig'])
    df = pd.get_dummies(df, columns=['LandContour'])
    df = pd.get_dummies(df, columns=['LotShape'])
    df = pd.get_dummies(df, columns=['Street'])
    df = pd.get_dummies(df, columns=['PavedDrive'])
    df = pd.get_dummies(df, columns=['SaleCondition'])

    # bin Fare into five intervals with equal amount of values
    # df['Fare-bin'] = pd.qcut(df['Fare'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # bin Age into seven intervals with equal amount of values
    # ('baby','child','teenager','young','mid-age','over-50','senior')
    # bins = [0, 4, 12, 18, 30, 50, 65, 100]
    # age_index = (1, 2, 3, 4, 5, 6, 7)
    # df['Age-bin'] = pd.cut(df['Age'], bins, labels=age_index).astype(int)

    # create a new column 'family' as a sum of 'SibSp' and 'Parch'
    # df['family'] = df['SibSp'] + df['Parch'] + 1
    # df['family'] = df['family'].map(lambda x: 4 if x > 4 else x)

    # create a new column 'FTicket' as the first character of the 'Ticket'
    # df['FTicket'] = df['Ticket'].map(lambda x: x[0])
    # combine smaller categories into one
    # df['FTicket'] = df['FTicket'].replace(['W', 'F', 'L', '5', '6', '7', '8', '9'], '4')
    # convert 'FTicket' values to new columns
    # df = pd.get_dummies(df, columns=['FTicket'])

    # get titles from the name
    # df['title'] = df.apply(lambda row: re.split('[,.]+', row['Name'])[1], axis=1)

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
    num_complete_minibatches = math.floor(set_size / mini_batch_size)

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
