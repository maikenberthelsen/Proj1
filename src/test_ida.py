import numpy as np
import matplotlib.pyplot as plt
from helpers import *

def plot_boxplot():
	a , input_data , b =load_csv_data('/Users/idasandsbraaten/Dropbox/Rolex/data/smallerTrainFixed.csv', sub_sample=True)
	plt.boxplot(input_data)
	#savefig('foo.png')
	#plt.show()

plot_boxplot()