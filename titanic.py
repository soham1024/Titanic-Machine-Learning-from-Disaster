import pandas as pd 
import numpy as np 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import statistics 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

df= pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')         		    #for test

"""
#					KNN Method               KNN Method 				Knn method			         KNN Method

#  													for train

#age correction
df['Age'].replace(np.nan,0, inplace= True)
mean_age= statistics.mean(df['Age'])
df['Age'].replace(0,mean_age, inplace=True)

df.replace('?',-99999, inplace= True)
df.drop(['PassengerId'], 1 , inplace=True)

df['Sex'].replace('male','1', inplace= True)
df['Sex'].replace('female','0', inplace= True)
df['Embarked'].replace('S','1' , inplace= True)
df['Embarked'].replace('C','2', inplace= True)
df['Embarked'].replace('Q','3', inplace= True)
df['Embarked'].replace(np.nan,'3', inplace= True)
#print(df['Embarked'].value_counts())

#												prediction for train

y=np.array(df['Survived'])
x=np.array(df[['Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']])

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier(10)
clf.fit(x_train,y_train)

accuracy= clf.score(x_test,y_test)
print(accuracy)

#                                                    for test

#													age correction
df_test['Age'].replace(np.nan,0, inplace= True)
mean_age_test= statistics.mean(df_test['Age'])
df_test['Age'].replace(0,mean_age_test, inplace=True)

#													sex correction
df_test['Sex'].replace('male','1', inplace= True)
df_test['Sex'].replace('female','0', inplace= True)
#													Embarked correctrion
df_test['Embarked'].replace('S','1' , inplace= True)
df_test['Embarked'].replace('C','2', inplace= True)
df_test['Embarked'].replace('Q','3', inplace= True)
df_test['Embarked'].replace(np.nan,'3', inplace= True)

#													fare correction
df_test['Fare'].replace(np.nan,0, inplace= True)
mean_Fare_test= statistics.mean(df_test['Fare'])
df_test['Fare'].replace(0,mean_Fare_test, inplace=True)

#print(df_test['Embarked'].value_counts())

#print(df_test.isna().sum())

#												prediction for test

example = np.array(df_test[['Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']])
example = example.reshape(len(example), -1)

prediction = clf.predict(example)
print(prediction)

kaggle_data = pd.DataFrame({'PassengerId':df_test['PassengerId'], 'Survived':prediction}).set_index('PassengerId')
kaggle_data.to_csv('sub.csv')

"""

# 	random forest method		random forest method		random forest method		random forest method


#  													for train


#age correction
df['Age'].replace(np.nan,0, inplace= True)
mean_age= statistics.mean(df['Age'])
df['Age'].replace(0,mean_age, inplace=True)

df.replace('?',-99999, inplace= True)
df.drop(['PassengerId'], 1 , inplace=True)

df['Sex'].replace('male','1', inplace= True)
df['Sex'].replace('female','0', inplace= True)
df['Embarked'].replace('S','1' , inplace= True)
df['Embarked'].replace('C','2', inplace= True)
df['Embarked'].replace('Q','3', inplace= True)
df['Embarked'].replace(np.nan,'3', inplace= True)
#print(df['Embarked'].value_counts())

#												prediction for train

y=np.array(df['Survived'])
x=np.array(df[['Pclass','Sex', 'Age', 'SibSp', 'Parch' ,'Fare', 'Embarked']])

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2)

model=RandomForestClassifier(n_estimators=1000)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

#                                                    for test

#													age correction
df_test['Age'].replace(np.nan,0, inplace= True)
mean_age_test= statistics.mean(df_test['Age'])
df_test['Age'].replace(0,mean_age_test, inplace=True)

#													sex correction
df_test['Sex'].replace('male','1', inplace= True)
df_test['Sex'].replace('female','0', inplace= True)
#													Embarked correctrion
df_test['Embarked'].replace('S','1' , inplace= True)
df_test['Embarked'].replace('C','2', inplace= True)
df_test['Embarked'].replace('Q','3', inplace= True)
df_test['Embarked'].replace(np.nan,'3', inplace= True)

#													fare correction
df_test['Fare'].replace(np.nan,0, inplace= True)
mean_Fare_test= statistics.mean(df_test['Fare'])
df_test['Fare'].replace(0,mean_Fare_test, inplace=True)

#print(df_test['Embarked'].value_counts())

#print(df_test.isna().sum())

#												prediction for test

example = np.array(df_test[['Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']])
example = example.reshape(len(example), -1)

prediction = model.predict(example)
print(prediction)

kaggle_data = pd.DataFrame({'PassengerId':df_test['PassengerId'], 'Survived':prediction}).set_index('PassengerId')
kaggle_data.to_csv('sub.csv')

