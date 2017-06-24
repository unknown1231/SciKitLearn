import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# we might perform an operation on both....so we combine them
combine = [train_data, test_data]

# print(train_data.head())

# print(train_data.columns.values)

# print(train_data.tail()) [#prints the number of rows...if n=5 then 5 rows are printed]

# dataframe.info give all the info of the data.
# print(train_data.info())
# print('-'*20)
# print(test_data.info())
# print(train_data.describe(include=['o'])).....not working for some reason!!

# Observing features which are necessary fo Survival
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# g = sns.FacetGrid(train_data, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

# hue is the variable that defines the subset of the data
# grid = sns.FacetGrid(train_data, col='Pclass', hue='Survived')
# grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=3, aspect=2)

# alpha is the transparency of the bar
# grid.map(plt.hist, 'Age', alpha=1, bins=20)
# grid.add_legend()
# plt.show()

# grid = sns.FacetGrid(train_data, row='Embarked', size=3, aspect=2)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=['r', 'g'])
# grid.add_legend()
# plt.show()

# grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=3, aspect=2)
# ci draws some sort of a line at the confidence interval
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=1, ci=None)
# grid.add_legend()
# plt.show()

# print("before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)

train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_data, test_data]

# print("after", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(train_data['Title'], train_data['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# as_index=True removes the index numbering
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# in the above line mean() basically does the mean of Mr,Mrs,...,etc

# start:
# the below lines basically mean that people with Mr should be put in a single list and so on...
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
# print((train_data.head()))
# end:

# now that we have fully made use of name feature....we can drop it
train_data = train_data.drop(['Name', 'PassengerId'], axis=1)
test_data = test_data.drop(['Name'], axis=1)
combine = [train_data, test_data]
# print(train_data.shape, test_data.shape)

# converting Sex feature into numbers of 1,0
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
# print(train_data.head())

# in the graph below the Age groups are shown According to the Pclasses and Sex(Female=1, Male=0)
# grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', size=3, aspect=2)
# grid.map(plt.hist, 'Age', alpha=1, bins=20)
# grid.add_legend()
# plt.show()

# create an empty array to store the guessed values of Pclass x Gender combinations
guess_ages = np.zeros((2, 3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_data = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_data.median()

            # converting floating age to nearest 0.5 age
            guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            # indexing Age, Sex, Pclass
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i, j]

    # converting floating point age into integer
    dataset['Age'] = dataset['Age'].astype(int)

# print(train_data.head())

# age bands for correlation with survived
train_data['AgeBands'] = pd.cut(train_data['Age'], 5)
train_data[['AgeBands', 'Survived']].groupby('AgeBands', as_index=False).mean().sort_values(by='AgeBands', ascending=True)

# replacing with ordinals for Age based on AgeBands

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64), 'Age']

# print(train_data.head())

# remove the AgeBand feature as its not necessary now
train_data = train_data.drop(['AgeBands'], axis=1)
combine = [train_data, test_data]
# print(train_data.head())

# now... combine parch and SibSp
for dataset in combine:
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp']

train_data[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='FamilySize', ascending=False)

# now we create isAlone
for dataset in combine:
    dataset['isAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1
train_data[['isAlone', 'Survived']].groupby('isAlone', as_index=False).mean()

# now...drop FamilySize, Parch and SibSp
train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_data, test_data]
# print(train_data.head())
# print('\n\n\n')
# print(test_data.head())

# combine age and Pclass
for dataset in combine:
    dataset['AgeClass'] = dataset['Age'] * dataset['Pclass']
# [:row, :col]
train_data.loc[:, ['Age', 'Pclass', 'AgeClass']].head(10)

freq_port = train_data.Embarked.dropna().mode()[0]
# print(freq_port)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_data[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)

# convert Embarked feature into numerical
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# print(train_data.head())

# filling single value for Fare in test_data
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)
# print(test_data.head())

train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
train_data[['FareBand', 'Survived']].groupby('FareBand', as_index=False).mean().sort_values(by='FareBand', ascending=True)

# converting Fare values to Ordinal based on FareBand
for dataset in combine:
    dataset.loc[(dataset['Fare'] <= 7.91), 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31.0, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
# dropping FareBand
train_data = train_data.drop(['FareBand'], axis=1)
combine = [train_data, test_data]

# print(train_data.head(10))
# print(test_data.head(10))

# test and train
x_train = train_data.drop(['Survived'], axis=1)
y_train = train_data['Survived']

x_test = test_data.drop(['PassengerId'], axis=1).copy()
# x_train.shape, y_train.shape, x_test.shape

print("1. Logistic Regression\n2.SVM\n3.Perceptron\n4.Naives Bayes Classifier\n5.Decision Tree\n6.Random Forrest"
      "\n7.k-Nearest Neighbours\n")
ask = input('Enter your choice:')
if ask == '1':
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    log_predict = logreg.predict(x_test)
    accuracy = logreg.score(x_train, y_train)
    print('LogisticRegression accuracy is:', accuracy * 100)

if ask == '2':
    svc = SVC()
    svc.fit(x_train, y_train)
    svc_predict = svc.predict(x_test)
    accuracy = svc.score(x_train, y_train)
    print('Support Vector Machine accuracy is:', accuracy * 100)

if ask == '3':
    perceptron_train = Perceptron()
    perceptron_train.fit(x_train, y_train)
    perceptron_predict = perceptron_train.predict(x_test)
    accuracy = perceptron_train.score(x_train, y_train)
    print('Perceptron accuracy is:', accuracy * 100)

if ask == '4':
    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    gaussian_predict = gaussian.predict(x_test)
    accuracy = gaussian.score(x_train, y_train)
    print('Naive Bayes accuracy is:', accuracy * 100)

if ask == '5':
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)
    tree_predict = decision_tree.predict(x_test)
    accuracy = decision_tree.score(x_train, y_train)
    print('Decision Tree accuracy is:', accuracy * 100)

if ask == '6':
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    random_forest_predict = random_forest.predict(x_test)
    accuracy = random_forest.score(x_train, y_train)
    print('Random Forest accuracy is:', accuracy * 100)

if ask == '7':
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    knn_predict = knn.predict(x_test)
    accuracy = knn.score(x_train, y_train)
    print('k-NearestNeighbours accuracy is:', accuracy * 100)

# table to show all models and their accuracies
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Accuracy': [accuracy * 100, accuracy * 100, accuracy * 100,
              accuracy * 100, accuracy * 100, accuracy * 100,
              accuracy * 100, accuracy * 100, accuracy * 100]})

print('\n', '-' * 50, '\n\t\t\t\tMODEL ACCURACIES\t\t\t\t', '\n', '-' * 50)
print(models.sort_values(by='Accuracy', ascending=False))
