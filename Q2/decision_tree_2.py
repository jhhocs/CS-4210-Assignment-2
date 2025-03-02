#-------------------------------------------------------------------------
# AUTHOR: Joshua Ho
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train, test, and output the performance of the 3 models created by using each training set on the test set provided
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Variables used to ensure features are maped to the same numbers across all training instances 
first1 = True
first2 = True
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    features = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    num_features = len(row) - 1
    if first1:
        d = [dict() for x in range(num_features)]
    for i, row in enumerate(dbTraining):
        temp = []
        for x in range(num_features):
            if first1 and row[x] not in d[x]:
                d[x][row[x]] = len(d[x])
            temp.append(d[x].get(row[x]))
        X.append(temp)
    first1 = False
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    if first2:
        d2 = {}
    for row in dbTraining:
        if first2 and row[-1] not in d2:
            d2[row[-1]] = len(d2)
        Y.append(d2.get(row[-1]))
    first2 = False

    #Loop your training and test tasks 10 times here
    average_accuracy = 0
    for i in range (10):

        #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        #--> add your Python code here
        dbTestTraining = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTestTraining.append(row)
                    # print(row)
                else:
                    features = row

        num_features = len(features) - 1

        dbTest = []
        correct = 0
        total = 0
        for i, row in enumerate(dbTestTraining):
            temp = []
            for x in range(num_features):
                if row[x] in d[x]:
                    temp.append(d[x].get(row[x]))
            temp.append(d2.get(row[x + 1]))
            dbTest.append(temp)

        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            class_predicted = clf.predict([[data[0], data[1], data[2], data[3]]])[0]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if class_predicted == data[4]:
                correct += 1
            total += 1

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
        average_accuracy += (correct / total)
    average_accuracy /= 10
    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {ds}: {average_accuracy}")