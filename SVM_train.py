import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


# load class 0
df_1 = pd.read_csv(filepath_or_buffer='vecs1.csv', header=None)
df_1.insert(256, "class", 0)

# load class 1
df_2 = pd.read_csv(filepath_or_buffer='vecs2.csv', header=None)
df_2.insert(256, "class", 1)

# concatenate Dataframes
df = pd.concat([df_1, df_2], axis=0)


# delete unecessary datafames
del df_1
del df_2

y = df.pop('class')
X = df

print(y)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print("x train")
print(X_train)
print("y train")
print(y_train)


###Creating Support Vector Machine Model
clf = svm.SVC()

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_predict))