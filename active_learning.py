import numpy as np
from sklearn.metrics import f1_score
from small_text.active_learner import PoolBasedActiveLearner
from small_text.classifiers import SklearnClassifierFactory


class ActiveLearner():
    def __init__(self, clf_template, query_strategy, train, num_classes=2, num_iterations=30, num_samples=500):
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.num_samples = num_samples

        self.train = train
        self.clf_template = clf_template
        self.query_strategy = query_strategy
        self.clf_factory = SklearnClassifierFactory(clf_template, self.num_classes)
        self.active_learner = PoolBasedActiveLearner(self.clf_factory, self.query_strategy, train)
        self.labeled_indices = self.initialize_strategy()

    def initialize_strategy(self):
        # Initialize the model. This is required for model-based query strategies.
        indices_pos_label = np.where(self.train.y == 1)[0]
        indices_neg_label = np.where(self.train.y == 0)[0]

        indices_initial = np.concatenate([np.random.choice(indices_pos_label, 100, replace=False),
                                          np.random.choice(indices_neg_label, 100, replace=False)],
                                         dtype=int)

        self.active_learner.initialize_data(indices_initial, self.train.y[indices_initial])

        return indices_initial

    def evaluate(self, active_learner, train, test):
        y_pred = active_learner.classifier.predict(train)
        y_pred_test = active_learner.classifier.predict(test)

        f1_score_train = f1_score(y_pred, train.y, average='micro')
        f1_score_test = f1_score(y_pred_test, test.y, average='micro')

        print('Train accuracy: {:.2f}'.format(f1_score_train))
        print('Test accuracy: {:.2f}'.format(f1_score_test))
        print('---')
        return f1_score_train, f1_score_test

    def perform_active_learning(self, test):
        """
        This is the main loop in which we perform num_iterations of active learning.
        During each iteration num_samples samples are queried according to the query strategy and then updated.
        The update step reveals the true label to the active learner, i.e. this is a simulation,
        but in a real scenario the user input would be passed to the update function.
        """
        # Set up array to track accuracy scores
        train_accuracy = []
        test_accuracy = []

        # Perform num_iterations of active learning...
        for i in range(self.num_iterations):
            # ...where each iteration consists of labelling num_samples using one
            # of the following strategies: Random Sampling, Least Confidence Sampling,
            # Prediction Entropy Sampling or Discriminative Active Learning
            indices_queried = self.active_learner.query(num_samples=self.num_samples)

            # Simulate user interaction here. Replace this for real-world usage.
            y = self.train.y[indices_queried]

            # Return the labels for the current query to the active learner.
            self.active_learner.update(y)

            self.labeled_indices = np.concatenate([indices_queried, self.labeled_indices])

            print('Iteration #{:d} ({} samples)'.format(i, len(self.labeled_indices)))
            train_acc, test_acc = self.evaluate(self.active_learner, self.train[self.labeled_indices], test)
            train_accuracy = np.append(train_accuracy, train_acc)
            test_accuracy = np.append(test_accuracy, test_acc)

        return train_accuracy, test_accuracy
