import matplotlib.pyplot as plt
import numpy as np
import re

def plot_accurasy(network_name):

	# ---- accurasy -----
	
	accurasy_train = extractAccurasy(network_name) 

	accurasy_file = open('records/' + network_name + '_' + 'validation' + '_accuracy.txt', 'r') 
	accurasy_all = accurasy_file.read()
	accurasy_validation = accurasy_all.split(';')
	accurasy_validation.remove('')
	for x in accurasy_validation:
		#print(x)
		x = float(x)	

	plt.title("ACCURASY grafik za mrezu " + network_name)	
	plt.plot(accurasy_validation, color = '#FF0000', linewidth = 1.0, linestyle = "-", label = 'validation')
	plt.plot(accurasy_train, color = '#0000FF', linewidth = 1.0, linestyle = "--", label = 'train')
	plt.legend(loc = 'upper right')
	plt.legend().draggable()
	#plt.show()
	plt.savefig('graphs/accurasy_' + network_name +'.png')
	plt.gcf().clear()

def plot_cost(network_name):

	# ----- accurasy -----
	cost_file = open('records/' + network_name + '_' + 'validation' + '_cost.txt', 'r') 
	cost_all = cost_file.read()
	cost_validation = cost_all.split(';')
	cost_validation.remove('')
	for x in cost_validation:
		#print(x)
		x = float(x)

	cost_file = open('records/' + network_name + '_' + 'train' + '_cost.txt', 'r') 
	cost_all = cost_file.read()
	cost_train = cost_all.split(';')
	cost_train.remove('')
	for x in cost_train:
		#print(x)
		x = float(x)	

	plt.title("COST grafik za mrezu " + network_name)	
	plt.plot(cost_validation, color = '#FF0000', linewidth = 1.0, linestyle = "-", label = 'validation')
	plt.plot(cost_train, color = '#0000FF', linewidth = 1.0, linestyle = "--", label = 'train')
	plt.legend(loc = 'upper right')
	plt.legend().draggable()
	#plt.show()
	plt.savefig('graphs/cost_' + network_name +'.png')
	plt.gcf().clear()

def extractAccurasy(network_name):
	# ---- accurasy train ----- 
	accurasy_array = []
	with open('records/' + network_name + '_description.txt', 'r') as accurasy_train_file:
		for line in accurasy_train_file:
			print(line)
			matchObj = re.match( r'Train.*Accuracy: ([0-9]+.[0-9]+).*', line)
			if matchObj:
				print("a")
				#match = re.search(regex,line)		
				print(matchObj.group(1))
				accurasy_array.append(matchObj.group(1))
			else:
				print("Match does not exist")		

	print(accurasy_array)
	return accurasy_array

def compareTwo(nname1,nname2,text):

	accurasy_file = open('records/' + nname1 + '_' + 'validation' + '_accuracy.txt', 'r') 
	accurasy_all = accurasy_file.read()
	accurasy_nn1 = accurasy_all.split(';')
	accurasy_nn1.remove('')
	for x in accurasy_nn1:
		#print(x)
		x = float(x)

	accurasy_file = open('records/' + nname2 + '_' + 'validation' + '_accuracy.txt', 'r') 
	accurasy_all = accurasy_file.read()
	accurasy_nn2 = accurasy_all.split(';')
	accurasy_nn2.remove('')
	for x in accurasy_nn2:
		#print(x)
		x = float(x)		

	plt.title(nname1 + ' i ' + nname2 + ' ' + text)	
	plt.plot(accurasy_nn1, color = '#FF0000', linewidth = 1.0, linestyle = "-", label = nname1)
	plt.plot(accurasy_nn2, color = '#0000FF', linewidth = 1.0, linestyle = "--", label = nname2)
	plt.legend(loc = 'upper right')
	plt.legend().draggable()
	#plt.show()
	plt.savefig('compareGraphs/accurasy_' + nname1 + nname2 +'.png')
	plt.gcf().clear()

def compareAllLambda(networks,lambdas):
	i = 0
	for name in networks:
		accurasy_file = open('records/' + name + '_' + 'validation' + '_accuracy.txt', 'r') 
		accurasy_all = accurasy_file.read()
		accurasy_nn = accurasy_all.split(';')
		accurasy_nn.remove('')
		for x in accurasy_nn:
			#print(x)
			x = float(x)
		plt.plot(accurasy_nn, color = '#FF0000', linewidth = 1.0, linestyle = "-", label = name + ' ' +str(lambdas[i]))
		i = i+1

	plt.title('Lambda compare')	
	plt.legend(loc = 'upper right')
	plt.legend().draggable()
	#plt.show()
	plt.savefig('compareGraphs/compareLambda.png')
	plt.gcf().clear()		

network_names = []
numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
for i in numbers:
	network_names.append('Network' + str(i))


#for network_name in network_names:
	#plot_accurasy(network_name)
	#plot_cost(network_name)

#print(network_names



#('Network1','Network3','Alpha 1.0 i 3.0')
#compareTwo('Network10','Network11','Lambda 0.001 i 0.0001')
#compareTwo('Network11','Network12','Lambda 0.0001 i 0.0005')
#compareTwo('Network15','Network16','Alpha 0.66 i 0.33')
#compareTwo('Network14','Network15','Epohe 50 i 30')
#compareTwo('Network1','Network2','reduce weights deviation true i false')
#compareTwo('Network7','Network10','Lambda 0.01 i 0.001')

compareAllLambda(['Network10','Network11','Network12','Network13','Network7'],[0.1,0.001,0.0005,0.0001,0.01])