from sklearn import tree
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class MyKNN():
	def fit(self,TrainingData, TrainingTarget):
		self.TrainingData = TrainingData;
		self.TrainingTarget = TrainingTarget;

	def predict(self,TestData):
		predictions = [];
		for eachrow in TestData:
			label = self.closest(eachrow);
			predictions.append(label);
		return predictions;

	def closest(self,eachrow):
		bestdistance = euc(eachrow,self.TrainingData[0]);
		bestindex = 0;

		for i in range(1, len(self.TrainingData)):
			dist = euc(eachrow,self.TrainingData[i]);

			if dist < bestdistance:
				bestdistance = dist;
				bestindex = i;
		return self.TrainingTarget[bestindex];

def euc(a,b):
	return distance.euclidean(a,b);

def MyKNeighbor():
	border = '-'*60;

	iris = load_iris();

	data = iris.data;
	target = iris.target;

	print(border);
	print("Actual Dataset");
	print(border);

	for i in range(len(iris.target)):
		print("ID: %d		Label: %s		Feature: %s" %(i,iris.data[i],iris.target[i]));
	print("Size of Dataset: %d" %(i+1))

	data_train, data_test, target_train, target_test = train_test_split(data,target,test_size=0.35)

	print(border);
	print("Training Dataset");
	print(border);
	for i in range(len(data_train)):
		print("ID: %d		Label: %s		Feature: %s" %(i+1,data_train[i],target_train[i]));
	print("Size of Training Dataset: %d" %(i+1))
	
	
	print(border);
	print("Testing Dataset");
	print(border);
	for i in range(len(data_test)):
		print("ID: %d		Label: %s		Feature: %s" %(i+1,data_test[i],target_test[i]));
	print("Size of Training Dataset: %d" %(i+1))
	print(border);


	clf = MyKNN();
	clf.fit(data_train,target_train);
	predictions = clf.predict(data_test);
	Accuracy = accuracy_score(target_test,predictions);

	return Accuracy;

def main():

	Accuracy = MyKNeighbor();
	print('Accuracy with KNN is: ',(Accuracy*100),'%');

if __name__ == "__main__":
	main();
