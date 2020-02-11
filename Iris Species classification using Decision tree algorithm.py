import numpy as np;
from sklearn import tree;
from sklearn.datasets import load_iris;


iris = load_iris();

print("Features :");
print(iris.feature_names);

print("Target :");
print(iris.target_names);

test_index = [1,51,101];

#TRAIN
train_target = np.delete(iris.target,test_index); #train_target has 147x1 targets
train_data = np.delete(iris.data,test_index,axis=0);#train_data has data of 147x4 flowers 

#TEST
test_target = iris.target[test_index];#test_target has 3x1 targets
test_data = iris.data[test_index];#test_data has 3x4 data

#decide Algo
clf = tree.DecisionTreeClassifier();

#train Data
clf.fit(train_data,train_target);


print("values for testing");
print(test_target);

#test DATA
rslt = clf.predict(test_data);

print("After testing: ",rslt);


#Visuals
from sklearn.externals.six import StringIO;
import pydot;

dot_data = StringIO();

tree.export_graphviz(clf,out_file=dot_data,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True,impurity=False);
graph  = pydot.graph_from_dot_data(dot_data.getvalue());
graph[0].write_pdf("ML.pdf");







