# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 22:40:32 2021
@author : Nikhil J
@github : ML-Nikhil

Problem Statement : Applying DecisionTree Algorithm on "titanic" dataset, for 
Survived and enlist the conclusion
"""
# Import Libraries
import pandas as pd
import seaborn as sns
import statistics as ss
import matplotlib.pyplot as plt
df = pd.read_csv('titanic.csv')
df.describe(include='all')
df.info()
df.index
#  Categorcal Variables : pclass, embarked, sex, survived
#  Quantitative Variables : age, fare, sibsp, parch but age and fare are in object
#  Alphanumeric Variables : Names, Ticket

# 1. AGE : dtype object : has some ? values : it should be in float/int which is not
# change the datatype and replace ? values as NaN
df['age'] = pd.to_numeric(df['age'], errors = 'coerce')
print('The NaN Values in Age before imputation',df['age'].isna().sum())
df.isna().sum()
df['age'].fillna(df['age'].mean(),inplace=True)
df['age'].plot.hist(alpha = 0.5,edgecolor = 'black', title = 'mean ='+ str(ss.mean(df['age'])))

# 2. FARE
df.head(10)
# fare also has ? values and datatype is object. 
# replacing ? by NaN and object datatype as float
df['fare']=pd.to_numeric(df['fare'],errors='coerce')
df['fare'].isna().sum()
df.dropna(subset = ['fare'],inplace =True)
df['fare'].plot.hist(alpha = 0.5,edgecolor = 'black')

# Categorical Variables

# 1. Sex - Male and Female
sns.countplot(x="sex",hue="survived",data=df)
s_dummy = pd.get_dummies(df['sex'],drop_first=True)

# 2. pclass
sns.countplot(x="pclass",hue="survived",data=df)
p_dummy = pd.get_dummies(df['pclass'],drop_first=True)

# 3. Embarked : has some ? value replace it by mode
print('The common class in embarked :',ss.mode(df['embarked']))
# replacing the ? value by S
df['embarked']=df['embarked'].replace(({'?': 'S'}))
sns.countplot(x="embarked",hue="survived",data=df)
emb_dummy = pd.get_dummies(df['embarked'],drop_first=True)


# Concatenating dataset

df = pd.concat([df,s_dummy,p_dummy,emb_dummy],axis =1)

df.drop(['Passenger_id','pclass','name','ticket','embarked','sex'],axis =1,inplace=True,)

# Splitting Data
x = df.drop('survived',axis =1)
y = df['survived']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=4,min_samples_leaf=5)
RF_classifier.fit(x_train,y_train)

RF_classifier.feature_importances_

# Important features :Male,Fare,pclass2,parch,sibsp,age,embarked

# Selecting feature for tree
x1 = x.iloc[:,[0,1,3,4,6,8]]
y1 = y

x1_train, x1_test, y1_train,y1_test = train_test_split(x1,y1,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
DCT_classifier=DecisionTreeClassifier(criterion='entropy',max_depth =3,min_samples_leaf=5)
DCT_classifier.fit(x1_train,y1_train)
y1_pred = DCT_classifier.predict(x1_test)
confusion_matrix(y1_test,y1_pred)
print('The accuracy score',round(accuracy_score(y1_test,y1_pred)*100,2))
from sklearn import tree
tree.plot_tree(DCT_classifier)
#___Tree Plot__
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
cn=['0','1']
tree.plot_tree(DCT_classifier,class_names=cn,filled = True)

#Graphiz Store the GRAPH where the directory is set
from sklearn.tree import export_graphviz
dot_data = tree.export_graphviz(DCT_classifier,feature_names=['age','sibsp','fare','male','p-class3','S'] ,class_names=['0','1'],filled=True)
import graphviz
graph = graphviz.Source(dot_data)
graph.format = "png"
graph.render("DecisionTree_Titanic")
#_
# Conclusion : Max Depth :3
# Female : Pclass 1,2 :fare less or euqal to $31.68 : Survived
# Female : Pclass 3   :fare less or equal to $23.35 : Survived
# Male   : Age less or equal to 13.5 : sibsp less or 2: Survives
# Male   : Age more than 13.5 : fare less than 26 : No Survival


















