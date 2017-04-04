import pandas as pd    
import numpy as np  
'''
data=pd.read_csv('../Banknote_Description/banknote_description.csv',names=["variance","skewness","kurtosis","entropy","class"])
print(data.head())
'''
#Note that the columns are the following:
#1. variance of Wavelet Transformed image (continuous).
#2. skewness of Wavelet Transformed image (continuous).
#3. kurtosis of Wavelet Transformed image (continuous).
#4. entropy of image (continuous).
#5. class (integer).

#Evaluating GINI index: The Gini index is the name of the cost function used to evaluate splits in the dataset.
def gini_index(groups, class_values):
	gini=0.0
	for value in class_values:
		for group in groups:
			size=len(group)
			proportion=[row[-1] for row in group].count(value)/float(size)
			gini=gini + (proportion*(1.0-proportion))
	return gini

#Testing the def gini_index
#print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
#print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

def test_split(index,value,dataset):
	left, right=list(), list()
	for row in dataset:
		if row[index]<value:
			left.append(row)
		else:
			right.append(row)
	return left, right

def get_split(dataset):
	class_values=list(set(row[-1] for row in dataset))
	b_index,b_value,b_score,b_groups=999,999,999,None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups=test_split(index,row[index],dataset)
			gini=gini_index(groups,class_values)
			if gini<b_score:
				b_index.b_value,b_score,b_groups=index, row[index], gini, groups
	return{'index':b_index, 'value':b_value, 'groups': b_groups}


dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
split = get_split(dataset)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
