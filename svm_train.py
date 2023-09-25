import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import argparse
import joblib


def parse_arguments():
    '''Parses in CLI arguments'''
    parser = argparse.ArgumentParser(
                    prog='svm_train.py',
                    description='A CLI tool for training a Support Vector Machine model for morphing detection.'+
                    'You must specify the files containing the feature vectors of each class (bona fide and morphed).\n'+
                    'Class labels: 0 -> bona fide, 1 -> morphed',
                    epilog='')

    
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

    # print("x train")
    # print(X_train)
    # print("y train")
    # print(y_train)


    # Creating Support Vector Machine Model
    clf = svm.SVC()

    print("Training SVM...")
    clf.fit(X_train, y_train)
    print("Model trained!")

    # save the model to disk # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    
    print(f"Saving model to file {args.output}")
    joblib.dump(clf, args.output)
    print("Model saved!")

    print("Performing test...")
    y_predict = clf.predict(X_test)
    print("Accuracy (with testing set):",metrics.accuracy_score(y_test, y_predict))



    # # load the model from disk
    # loaded_model = joblib.load('finalized_model.sav')


    # y_predict = loaded_model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(y_test, y_predict))





if __name__ == "__main__":
    main()
