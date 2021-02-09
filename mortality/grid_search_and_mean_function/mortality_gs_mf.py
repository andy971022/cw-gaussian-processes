import itertools
from cwgp.cwgp import CWGP
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import GPy
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as rmse


def gp_regression(x, y, params):
	kernel = GPy.kern.Matern32(1)

	ratio = 0.9
	length = len(x)
	train = int(ratio*length)
	train_up, train_low = train, train-train


	model = GPy.models.GPRegression(x[train_low:train_up], y[train_low:train_up], kernel=kernel)
	model.optimize()

	return model

def gp_wrapper(x, y, params):
	cwgp = {}
	df = pd.DataFrame()
	t_data = copy.deepcopy(data)
	for index,param in enumerate(params):
		cwgp[index] = {}
		for i,(t,d) in enumerate(param):
			cwgp_model = CWGP(t,n=d)
			cwgp_model_p = cwgp_model.fit(t_data).x
			t_data, t_data_d = cwgp_model.phi.comp_phi(cwgp_model_p,t_data)
			cwgp[index][i] = (cwgp_model)


	

def grid_search(estimator, x, y, params):
	c = params.pop("c",2)
	n = params.pop("n",[1,2,3])
	transformations = params.pop("transformations",["sa","sal"])

	cwgp_params = [transformations,n]

	params_product =  list(itertools.product(*cwgp_params))
	params_combination = []

	for param in itertools.permutations(params_product,c):
		if param not in params_combination:
			params_combination.append(param)

	print(params_combination)
	estimator(y, params_combination)


if __name__ == '__main__':
	CSV_FIlE = "../japan3.csv"
	df = pd.read_csv(CSV_FIlE)
	age = 90
	df_all = {}
	df_all[age]= df[(df["age"]==age)]
	
	rate = df_all[age]["rate"].to_numpy().reshape(-1,1)
	year = df_all[age]["year"].to_numpy().reshape(-1,1)

	grid_search(gp_wrapper, year, rate,{})