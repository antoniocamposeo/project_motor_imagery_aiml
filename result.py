from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns


def result(train_score, test_score, validation_score, y_test, y_pred):
    if len(train_score) == 2:
        print(f"-train score:loss [{train_score[0]}],accuracy[{train_score[1]}]\n "
              f"-test score:loss [{test_score[0]}],accuracy[{test_score[1]}]\n"
              f"-validation score: loss [{validation_score[0]}],accuracy[{validation_score[1]}]\n")
    else:
        print(f"-train score:[{train_score[1]}]\n "
              f"-test score:[{test_score[1]}]\n"
              f"-validation score: [{validation_score[1]}]\n")

    cm = confusion_matrix(y_test, y_pred)
    print('\n clasification report:\n', classification_report(y_test, y_pred))
    print('\n confussion matrix:\n', confusion_matrix(y_test, y_pred))
    sns.heatmap(cm, annot=True)
    plt.show()

# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('F1 score:', f1_score(y_test, y_pred,average='macro'))
# print('Recall:', recall_score(y_test, y_pred,average=None))
# print('Precision:', precision_score(y_test, y_pred))
