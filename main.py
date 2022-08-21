from locale import normalize

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from small_text.classifiers import ConfidenceEnhancedLinearSVC, SklearnClassifierFactory
from small_text.data import SklearnDataset
from small_text.query_strategies import RandomSampling, PoolExhaustedException, EmptyPoolException, LeastConfidence, PredictionEntropy, DiscriminativeActiveLearning

from active_learning import ActiveLearner
from plotter import Plotter


def main():
    train_file = 'DataFiles/train.csv'
    dataframe = pd.read_csv(train_file, sep=',')

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(dataframe['lyric'].to_list())

    train, test = preprocess_data(dataframe, tfidf_vectorizer)

    # Active learning parameters
    num_classes = 2
    num_iterations = 30
    clf_template = ConfidenceEnhancedLinearSVC()
    clf_factory = SklearnClassifierFactory(clf_template, num_classes)
    query_strategies = [RandomSampling(), LeastConfidence(), PredictionEntropy(), DiscriminativeActiveLearning(clf_factory, num_iterations=num_iterations)]
    strategy_names = ["Random Sampling", "Least Confidence Sampling", "Prediction Entropy Sampling", "Discriminative Active Learning"]

    train_dict, test_dict, active_learner_dict = run(clf_template, train, test, query_strategies, strategy_names)

    # Plot the accuracy rates for all the different models
    plotter = Plotter(train_dict, test_dict)
    plotter.plot_accuracies(num_iterations=num_iterations)

    # Predict a number of samples for each model
    val_file = 'DataFiles/test.csv'
    validation_dataframe = pd.read_csv(val_file, sep=',', nrows=15)
    val_x = normalize(tfidf_vectorizer.transform(validation_dataframe['lyric'].to_list()))
    val_y = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    val_data = SklearnDataset(val_x, val_y)

    for idx, active_learner in enumerate(active_learner_dict.values()):
        val_pred = active_learner.classifier.predict(val_data)
        name = [name for name,learner in active_learner_dict.items() if learner == active_learner]
        plotter.plot_confusion_matrix(val_y, val_pred, name)
        print()

def preprocess_data(df, vectorizer):
    X = df['lyric'].to_list()
    y = df['class'].to_numpy()

    x_train_split, x_test_split, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

    x_train = normalize(vectorizer.transform(x_train_split))
    x_test = normalize(vectorizer.transform(x_test_split))

    return SklearnDataset(x_train, y_train), SklearnDataset(x_test, y_test)


def run(clf_template, train, test, query_strategies, strategy_names):
    train_dict = {}
    test_dict = {}
    active_learner_dict = {}

    for i, query_strategy in enumerate(query_strategies):
        active_learner = ActiveLearner(clf_template, query_strategy, train)
        try:
            train_accuracy, test_accuracy = active_learner.perform_active_learning(test)
            active_learner_dict.update({strategy_names[i]: active_learner.active_learner})
            train_dict.update({strategy_names[i]: train_accuracy})
            test_dict.update({strategy_names[i]: test_accuracy})
        except PoolExhaustedException:
            print('Error! Not enough samples left to handle the query.')
        except EmptyPoolException:
            print('Error! No more samples left. (Unlabeled pool is empty)')

    return train_dict, test_dict, active_learner_dict


if __name__ == '__main__':
    main()
