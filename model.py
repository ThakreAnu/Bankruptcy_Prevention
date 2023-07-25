import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

bankrupt = pd.read_csv('bankruptcy-prevention.csv', sep = ';', header = 0)
print(bankrupt)

print(bankrupt.info())
print(bankrupt.shape)

bankrupt.isnull().sum()

bankrupt_new = bankrupt.iloc[:,:]
print(bankrupt_new)
bankrupt_new["class_yn"] = 1
bankrupt_new.loc[bankrupt[' class'] == 'bankruptcy', 'class_yn'] = 0

print(bankrupt_new)

bankrupt_new.drop(' class', inplace = True, axis =1)
print(bankrupt_new.head())


from sklearn.model_selection import train_test_split # trian and test
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

# Input
x = bankrupt_new.iloc[:,:-1]

# Target variable

y = bankrupt_new.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)





from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)

print(f"Training Accuracy : {dt.score(x_train,y_train)}")
print(f"Testing Accuracy : {dt.score(x_test,y_test)}")
print(classification_report(y_test,y_pred))


#make pickle
import pickle
pickle.dump(dt,open('model.pkl','wb'))