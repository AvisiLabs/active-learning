import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.metrics import f1_score
from small_text.data import SklearnDataset
from small_text.query_strategies import RandomSampling, LeastConfidence, PredictionEntropy, DiscriminativeActiveLearning
from small_text.active_learner import PoolBasedActiveLearner
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

train_file = 'DataFiles/train.csv'
dataframe = pd.read_csv(train_file, sep=',')

vectorizer = TfidfVectorizer(stop_words='english')


def preprocess_data(df, vectorizer):
    X = df['lyric'].to_list()
    y = df['class'].to_numpy()

    x_train_split, x_test_split, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

    x_train = normalize(vectorizer.fit_transform(x_train_split))
    x_test = normalize(vectorizer.transform(x_test_split))

    return SklearnDataset(x_train, y_train), SklearnDataset(x_test, y_test)


train, test = preprocess_data(dataframe, vectorizer)


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    f1_score_train = f1_score(y_pred, train.y, average='micro')
    f1_score_test = f1_score(y_pred_test, test.y, average='micro')

    print('Train accuracy: {:.2f}'.format(f1_score_train))
    print('Test accuracy: {:.2f}'.format(f1_score_test))
    print('---')
    return f1_score_train, f1_score_test


def perform_active_learning(active_learner, train, indices_labeled, test, num_iterations, num_samples):
    """
    This is the main loop in which we perform num_iterations of active learning.
    During each iteration num_samples samples are queried and then updated.
    The update step reveals the true label to the active learner, i.e. this is a simulation,
    but in a real scenario the user input would be passed to the update function.
    """
    # Set up array to track accuracy scores
    train_accuracy = []
    test_accuracy = []

    # Perform num_iterations of active learning...
    for i in range(num_iterations):
        # ...where each iteration consists of labelling num_samples
        indices_queried = active_learner.query(num_samples=num_samples)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print('Iteration #{:d} ({} samples)'.format(i, len(indices_labeled)))
        train_acc, test_acc = evaluate(active_learner, train[indices_labeled], test)
        train_accuracy = np.append(train_accuracy, train_acc)
        test_accuracy = np.append(test_accuracy, test_acc)

    return train_accuracy, test_accuracy


def initialize_strategy(active_learner, y_train):
    # Initialize the model. This is required for model-based query strategies.
    indices_pos_label = np.where(y_train == 1)[0]
    indices_neg_label = np.where(y_train == 0)[0]

    indices_initial = np.concatenate([np.random.choice(indices_pos_label, 100, replace=False),
                                      np.random.choice(indices_neg_label, 100, replace=False)],
                                     dtype=int)

    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


num_classes = 2
num_iterations = 30
num_samples = 500

# Active learning parameters
clf_template = ConfidenceEnhancedLinearSVC()
clf_factory = SklearnClassifierFactory(clf_template, num_classes)
query_strategy = RandomSampling()

# Active learner
active_learner_random = PoolBasedActiveLearner(clf_factory, query_strategy, train)
labeled_indices = initialize_strategy(active_learner_random, train.y)

try:
    train_accuracy_random, test_accuracy_random = perform_active_learning(active_learner_random, train, labeled_indices, test, num_iterations=num_iterations,
                                                                          num_samples=num_samples)
except PoolExhaustedException:
    print('Error! Not enough samples left to handle the query.')
except EmptyPoolException:
    print('Error! No more samples left. (Unlabeled pool is empty)')

# Active learning parameters
clf_template = ConfidenceEnhancedLinearSVC()
clf_factory = SklearnClassifierFactory(clf_template, num_classes)
query_strategy = LeastConfidence()

# Active learner
active_learner_confidence = PoolBasedActiveLearner(clf_factory, query_strategy, train)
labeled_indices = initialize_strategy(active_learner_confidence, train.y)

try:
    train_accuracy_confidence, test_accuracy_confidence = perform_active_learning(active_learner_confidence, train, labeled_indices, test, num_iterations=num_iterations,
                                                                                  num_samples=num_samples)
except PoolExhaustedException:
    print('Error! Not enough samples left to handle the query.')
except EmptyPoolException:
    print('Error! No more samples left. (Unlabeled pool is empty)')

# Active learning parameters
clf_template = ConfidenceEnhancedLinearSVC()
clf_factory = SklearnClassifierFactory(clf_template, num_classes)
query_strategy = PredictionEntropy()

# Active learner
active_learner_entropy = PoolBasedActiveLearner(clf_factory, query_strategy, train)
labeled_indices = initialize_strategy(active_learner_entropy, train.y)

try:
    train_accuracy_entropy, test_accuracy_entropy = perform_active_learning(active_learner_entropy, train, labeled_indices, test, num_iterations=num_iterations,
                                                                            num_samples=num_samples)
except PoolExhaustedException:
    print('Error! Not enough samples left to handle the query.')
except EmptyPoolException:
    print('Error! No more samples left. (Unlabeled pool is empty)')

# Active learning parameters
clf_template = ConfidenceEnhancedLinearSVC()
clf_factory = SklearnClassifierFactory(clf_template, num_classes)
query_strategy = DiscriminativeActiveLearning(clf_factory, num_iterations=num_iterations)

# Active learner
active_learner_dal = PoolBasedActiveLearner(clf_factory, query_strategy, train)
labeled_indices = initialize_strategy(active_learner_dal, train.y)

try:
    train_accuracy_dal, test_accuracy_dal = perform_active_learning(active_learner_dal, train, labeled_indices, test, num_iterations=num_iterations, num_samples=num_samples)
except PoolExhaustedException:
    print('Error! Not enough samples left to handle the query.')
except EmptyPoolException:
    print('Error! No more samples left. (Unlabeled pool is empty)')

# Plot the accuracy rates for all the different models
epochs = range(num_iterations)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 7), dpi=200)
axs[0, 0].plot(epochs, train_accuracy_random, 'g', label='Training accuracy')
axs[0, 0].plot(epochs, test_accuracy_random, 'b', label='Test accuracy')
axs[0, 0].legend()
axs[0, 0].set_title('Random Sampling')

axs[0, 1].plot(epochs, train_accuracy_confidence, 'g', label='Training accuracy')
axs[0, 1].plot(epochs, test_accuracy_confidence, 'b', label='Test accuracy')
axs[0, 1].set_title('Least Confidence Sampling')

axs[1, 0].plot(epochs, train_accuracy_entropy, 'g', label='Training accuracy')
axs[1, 0].plot(epochs, test_accuracy_entropy, 'b', label='Test accuracy')
axs[1, 0].set_title('Entropy Sampling')

axs[1, 1].plot(epochs, train_accuracy_dal, 'g', label='Training accuracy')
axs[1, 1].plot(epochs, test_accuracy_dal, 'b', label='Test accuracy')
axs[1, 1].set_title('Discriminative Active Learning')

fig.suptitle('Training and Test accuracy')
fig.supxlabel('Epochs')
fig.supylabel('Accuracy')
plt.show()
