#-------------------------------------------------------------------------
# AUTHOR: Joshua Ho
# FILENAME: knn.py
# SPECIFICATION: Compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
error = 0
total = 0

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Loop your data to allow each instance to be your test set
for i, row in enumerate(db):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    num_features = len(row) - 1
    for i2, row2 in enumerate(db):
        temp = []
        if i != i2:
            for j2 in range(num_features):
                temp.append(float(row2[j2]))
            X.append(temp)

    
    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for i2, row2 in enumerate(db):
        if i != i2:
            if row2[-1] == "ham":
                Y.append(0)
            else:
                Y.append(1)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    test_sample = []
    for j in range(num_features):
        test_sample.append(float(row[j]))

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([test_sample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    expected = 0

    if row[-1] == "spam":
        expected = 1
    if class_predicted != expected:
        error += 1
    total += 1


#Print the error rate
#--> add your Python code here
print(f"Error: {error}, Total: {total}")
print(f"Error rate: {error / total}")





