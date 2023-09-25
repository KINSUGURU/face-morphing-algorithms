import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
    '''Parses in CLI arguments'''
    parser = argparse.ArgumentParser(
                    prog='svm_train.py',
                    description='A CLI tool for training a Support Vector Machine model for morphing detection.'+
                    'You must specify the files containing the feature vectors of each class (bona fide and morphed).\n'+
                    'Class labels: 0 -> bona fide, 1 -> morphed',
                    epilog='')


    parser.add_argument('-v', '--visualize', action='store_true', help='Plot graphs for the dataset and Confusion matrix')  # on/off flag
    
    requiredArgs = parser.add_argument_group('Required arguments')

    requiredArgs.add_argument('-b', '--bonafide', type=argparse.FileType('r'), nargs='+', help='Provide the .csv file(s) with bona fide vectors.', required=True)

    requiredArgs.add_argument('-m', '--morphed', type=argparse.FileType('r'), nargs='+', help='Provide the .csv file(s) with morphed vectors.', required=True)

    requiredArgs.add_argument('-o', '--output', help='Name of file that will save the trained model (e.g. model.sav).', required=True)
    
    return parser.parse_args()



def main():

    args = parse_arguments()

    data_frames = []

    # load class 0
    for bona_fide_csv in args.bonafide:

        df_1 = pd.read_csv(filepath_or_buffer=bona_fide_csv, header=None)
        df_1.insert(256, "class", 0)
        data_frames.append(df_1)

    # load class 1
    for morphed_csv in args.morphed:

        df_1 = pd.read_csv(filepath_or_buffer=morphed_csv, header=None)
        df_1.insert(256, "class", 1)
        data_frames.append(df_1)


    # concatenate Dataframes
    df = pd.concat(data_frames, axis=0)

    # delete unecessary datafames
    del data_frames

    y = df.pop('class')
    X = df

    # print(y)
    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=23)

    print("x train")
    print(X_train)
    print("y train")
    print(y_train)


    # Creating Support Vector Machine Model
    clf = svm.SVC()

    print("Training SVM...")
    clf.fit(X_train, y_train)
    print("Model trained!")

    # save the model to disk # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    
    print(f"Saving model to file {args.output}")
    joblib.dump(clf, args.output)
    print("Model saved!")

    print("Performing tests...")

    y_predict = clf.predict(X_test)
    print("Accuracy (with testing set):",metrics.accuracy_score(y_test, y_predict))

    # confusion matrix
    cm_test = confusion_matrix(y_test, y_predict)#labels=['bona fide', 'morphed']
    print("Confusion Matrix on Testing Set")
    print(cm_test)


    y_predict = clf.predict(X_train)
    print("Accuracy (with training set):",metrics.accuracy_score(y_train, y_predict))

    # confusion matrix
    cm_train = confusion_matrix(y_train, y_predict)#labels=['bona fide', 'morphed']
    print("Confusion Matrix on Training Set")
    print(cm_train)



    if args.visualize:


        # Pie charts
        # https://www.w3schools.com/python/matplotlib_pie_charts.asp
        
        num_train_0 = len(y_train[y_train[:]==0])
        num_train_1 = len(y_train[y_train[:]==1])
        pie_train = np.array([num_train_0, num_train_1])
        labels_train = [f"bona fide: {num_train_0}", f"morphed: {num_train_1}"]

        num_test_0 = len(y_test[y_test[:]==0])
        num_test_1 = len(y_test[y_test[:]==1])
        pie_test = np.array([num_test_0, num_test_1])
        labels_test = [f"bona fide: {num_test_0}", f"morphed: {num_test_1}"]


        # https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/figure_title.html
        fig, axs = plt.subplots(1, 2, constrained_layout=True)


        fig.suptitle('Dataset Overview', fontsize=16)

        axs[0].pie(pie_train, labels=labels_train, shadow=True, explode=[0.1, 0])
        axs[0].set_title('Training set')
        axs[0].legend(title = "Training set:")

        axs[1].pie(pie_test, labels=labels_test, shadow=True, explode=[0.1, 0])
        axs[1].set_title('Testing Set')
        axs[1].legend(title = "Testing set:")


        disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        disp.plot()
        plt.title("Confusion Matrix on Testing Set")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_train)
        disp.plot()
        plt.title("Confusion Matrix on Training Set")


        plt.show()




    # # load the model from disk
    # loaded_model = joblib.load('finalized_model.sav')


    # y_predict = loaded_model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(y_test, y_predict))





if __name__ == "__main__":
    main()
