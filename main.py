from sklearn import tree

# [height,weight,shoesize]
x = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],
     [171,75,42],[181,85,43]]
y = ['male','female','female','female','male','male','male','female','male','female','male']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

prediction = clf.predict([[190,70,43]])

print (prediction)



#using KNN classifiers (K Nearest Neighbors classifier)

#actual tool to use KNN classifier
from sklearn.neighbors import KNeighborsClassifier




# the following are used to test accuracy

from sklearn.metrics import accuracy_score


classifier = KNeighborsClassifier(n_neighbors=5)
classifier = classifier.fit (x,y)
prediction_1 = classifier.predict([[190,70,43]])

print(prediction_1)