import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib #model persistance
from sklearn import tree #decision tree

from flask import Flask

data = pd.read_csv(r'C:\Users\Akiva Goldberg\Documents\Titanic Project\titanic.csv')

#prepare data - data is fine, no irrelevances or incomplete data
# 70-80% for training, rest for testing
#split data - split into input set (age, gender), and output set (genre) for training
Input = data.drop(columns=['Survived'])
Output = data['Survived']
Intrain, Intest, Outtrain, Outtest = train_test_split(Input, Output, test_size=0.25) #20% data for testing

#Create model
model = MLPClassifier()
model.fit(Intrain, Outtrain)

joblib.dump(model, 'music-recommender.joblib') # #Model Persistance - save a trained model so you don't need to train it again
model = joblib.load('music-recommender.joblib')

#Predict
predictions = model.predict(Intest) 

#Evaluate
score = accuracy_score(Outtest, predictions)
print (score)

#Visualize tree
#tree.export_graphviz(model, out_file='music-recommender.dot', 
#                    feature_names=['age', 'gender'], 
#                    class_names=sorted(Output.unique()), #unique list of classes
 #                   label='all',
  #                  rounded=True,
   #                 filled=True)
medianSal = 61000
medianFare = 14.45
age = input("Age:")
gender = input("Male or Female:")
if (gender=="Male" or gender=="male"):
    male = 1
else:
    male = 0
sibs = input("Siblings/Spouses Aboard:")
parents = input("Parents/Children Aboard:")
salary = int(input("Salary:"))
fare = salary/(medianSal/medianFare)
if (fare < medianFare):
    pclass = 3
elif (fare > 25):
    pclass = 1
else:
    pclass = 2

print (model.predict([[pclass, age, male, sibs, parents, salary]]))


