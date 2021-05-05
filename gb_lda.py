import pandas as pd
import os
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def check_difference(s):
    if s['prediction'] != s['label']:
        return '+'


def sk_classification(path_features, path_info, path_save):

    """
        Experiment B classification using sklearn
        :param path_features: path to folder containing feature files of single subjects
        :param path_info: path to folder containing indicator file for all subjects
        :param path_save: path to save results
        :return:
        """

    # set of feature files in path_features folder, excluding subfolders
    ft_dfs = [pd.read_csv(path_features + str(el)) for el in os.listdir(path_features) if os.path.isfile(os.path.join(path_features, el))]

    df_predictions = pd.DataFrame(columns=['indicator', 'image', 'label', 'prediction', 'different'])
    indicator_column = pd.read_csv(path_info, usecols=['indicator'])
    image_column = pd.read_csv(path_info, usecols=['image'])

    pred_list = list()
    label_list = list()

    sys.stdout = open(path_save + 'sk_results.csv', 'w')

    for i in range(len(ft_dfs)):

        # list of features files copy
        training_list = ft_dfs.copy()

        # delete features with nan elements
        for df in training_list:
            df.dropna(axis=0, inplace=True)

        # removing test features
        training_list.pop(i)

        # creating features dataframe
        training_set = pd.concat(training_list, ignore_index=True)

        # unlabeled data for training set
        x_train = training_set.iloc[:, 0:-1]
        # labeled data for training set
        y_train = training_set['classes']
        # unlabeled data for test set
        x_test = ft_dfs[i].iloc[:, 0:-1]
        # labeled data for test set
        y_test = ft_dfs[i]['classes']

        # scaling data
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Classifier
        clf = GradientBoostingClassifier(random_state=0)
        # clf = AdaBoostClassifier(random_state=0)
        # clf = LinearDiscriminantAnalysis()
        # clf = RandomForestClassifier(random_state=0)
        # clf = VotingClassifier(estimators=[('GBC', GradientBoostingClassifier(random_state=0)), ('RF', RandomForestClassifier(random_state=0)), ('LDA', LinearDiscriminantAnalysis())])

        # fitting classifier
        clf.fit(x_train, y_train)

        # calculating predictions
        predictions = clf.predict(x_test)

        print('**************************************************************************')
        print('Dataset test: {}'.format(i))
        print("Accuracy score: {0:.3f}".format(clf.score(x_test, y_test)))

        print("\nConfusion Matrix:\n")
        print(confusion_matrix(y_test, predictions))

        print("\nClassification Report")
        print(classification_report(y_test, predictions))

        pred_list = pred_list + predictions.tolist()
        label_list = label_list + y_test.tolist()

    df_predictions['prediction'] = pred_list
    df_predictions['label'] = label_list
    df_predictions['indicator'] = indicator_column
    df_predictions['image'] = image_column
    df_predictions['different'] = df_predictions.apply(check_difference, axis=1)

    print("\nNumero totale: ")
    print(df_predictions[['image', 'indicator']].groupby(['indicator']).count())
    print("Numero errori: ")
    print(df_predictions[['different', 'indicator']].groupby(['indicator']).count())
    sys.stdout.close()



