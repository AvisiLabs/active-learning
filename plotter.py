from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Plotter:
    def __init__(self, train_accuracy_dict, test_accuracy_dict):
        self.train_accuracy_dict = train_accuracy_dict
        self.test_accuracy_dict = test_accuracy_dict

    def plot_accuracies(self, num_iterations):
        # Plot the accuracy rates for all the different models
        epochs = range(num_iterations)

        rows = round(len(self.train_accuracy_dict)/2)
        cols = len(self.train_accuracy_dict)-rows
        fig, axs = plt.subplots(rows, cols, sharex='all', sharey='all', figsize=(15, 7), dpi=200)

        for n in range(0, rows):
            for m in range(0,cols):
                axs[n,m].plot(epochs, list(self.train_accuracy_dict.values())[n+m], 'g', label='Training accuracy')
                axs[n,m].plot(epochs, list(self.test_accuracy_dict.values())[n+m], 'b', label='Test accuracy')
                axs[n,m].set_title(list(self.train_accuracy_dict)[n+m])

        fig.suptitle('Training and Test accuracy')
        fig.supxlabel('Epochs')
        fig.supylabel('Accuracy')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(targets, predictions, name):
        cm = confusion_matrix(targets, predictions, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
        disp.plot()
        plt.title(name)
        plt.show()
