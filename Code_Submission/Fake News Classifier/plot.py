import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

def create_plot(history, model_name):
    plt.title('Accuracy and Loss (' + model_name + ')')
    plt.plot(history.history['acc'], color='g', label='Train Accuracy')
    plt.plot(history.history['val_acc'], color='b', label='Validation Accuracy')
    plt.plot(history.history['loss'], color='r', label='Train Loss')
    plt.plot(history.history['val_loss'], color='m', label='Validation Loss')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("plot.eps")
    plt.savefig("plot.png")
    plt.savefig("plot.pdf")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig("confusion_matrix.png")
    plt.savefig("confusion_matrix.eps")
    plt.savefig("confusion_matrix.pdf")