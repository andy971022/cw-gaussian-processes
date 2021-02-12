import itertools
from cwgp.grid_search import grid_search
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import GPy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize
import os




def zero(x):
	return 0

def one(x):
	return 1

def average(x):
	return avg

def sigmoid(x):
	return  1/(1+np.exp(-x))

def exponential(x):
	return np.exp(-x)



def gp_regression(x, y, mean_func):
	# kernel = GPy.kern.RBF(1)
	kernel = GPy.kern.Matern32(1)

	mf = GPy.core.Mapping(1,1)
	mf.f = lambda x: mean_func(x)
	mf.update_gradients = lambda a,b: None


	model = GPy.models.GPRegression(x[train_low:train_up], y[train_low:train_up], mean_function=mf, kernel=kernel)
	model.optimize()

	y_pred = model.predict(x[train_up:])[0]
	y_true = y[train_up:]

	rmse = mean_squared_error(y_true, y_pred, squared=False)
	mae = mean_absolute_error(y_true, y_pred)

	txt = f"""rmse : {rmse} \n mae : {mae}"""

	model.plot(plot_limits=[1940,2031])

	plt.title('Age ' + str(age) + f" {mean_func.__name__}")
	plt.xlabel('Year')
	plt.ylabel('Mortality Rate' )
	plt.scatter(x[train_up:], y[train_up:], marker="x", color='red')
	plt.figtext(0.5, 0.2, txt, wrap=True, horizontalalignment='center', fontsize=12)
	# plt.ylim([0.1,0.35])
	plt.ylim([0,0.12])

	plt.grid(True)
	plt.savefig(f'./{IMG_DIR}/{mean_func.__name__}_{age}_mf.png')
	plt.show()


	return rmse, mae



if __name__ == '__main__':
	CSV_FIlE = "../japan3.csv"
	IMG_DIR = "./mf"

	df = pd.read_csv(CSV_FIlE)
	age = 75
	df_all = {}
	df_all[age]= df[(df["age"]==age)]

	rate = df_all[age]["rate"].to_numpy().reshape(-1,1)
	year = df_all[age]["year"].to_numpy().reshape(-1,1)

	ratio = 0.9
	length = len(rate)
	train = int(ratio*length)
	train_up, train_low = train, train-train
	

	avg = np.mean(rate[train_low:train_up])

	mean_funcs = [zero,one,sigmoid,exponential,average]

	if not os.path.exists(IMG_DIR):
		os.makedirs(IMG_DIR)

	for mf in mean_funcs:
		out = gp_regression(year, rate, mf)
		print(out)