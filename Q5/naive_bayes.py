#-------------------------------------------------------------------------
# AUTHOR: Joshua Ho
# FILENAME: naive_bayes.py
# SPECIFICATION: Output the classification of each of the 10 instances from the file weather_test if the classification confidence is >= 0.75
# FOR: CS 4210- Assignment #2 
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
db = []
X = []
Y = []

with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
    
#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
num_features = len(row) - 1

d = [dict() for x in range(num_features)]
for i, row in enumerate(db):
    temp = []
    for x in range(num_features):
        if x == 0:
            continue
        if row[x] not in d[x]:
            d[x][row[x]] = len(d[x])
        temp.append(d[x].get(row[x]))
    X.append(temp)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# d2 = {}
d2 = []
for row in db:
    if row[-1] not in d2:
        d2.append(row[-1])
        # d2[row[-1]] = len(d2)
    Y.append(d2.index(row[-1]))
#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
dbTest = []
header = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append(row)
        else:
            header = row
header.append("Confidence")

XTest = []
YTest = []
for i, row in enumerate(dbTest):
    temp = []
    for x in range(num_features):
        if x == 0:
            continue
        temp.append(d[x].get(row[x]))
    XTest.append(temp)
#Printing the header os the solution
#--> add your Python code here

print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(*header))

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for i, row in enumerate(dbTest):
    classification = None
    confidence = 0
    prediction = clf.predict_proba([XTest[i]])[0]
    if prediction[0] >= 0.75:
        classification = d2[0]
        confidence = round(prediction[0], 3)
    elif prediction[1] >= 0.75:
        classification = d2[1]
        confidence = round(prediction[1], 3)
    else:
        continue
    row.pop()
    row.append(classification)
    row.append(confidence)
    print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(*row))



