import itertools
from cwgp.cwgp import CWGP
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import GPy
from sklearn.metrics import mean_absolute_error, mean_squared_error





def gp_regression(x, y, cwgp_model):
	kernel = GPy.kern.Matern32(1)

	ratio = 0.9
	length = len(x)
	train = int(ratio*length)
	train_up, train_low = train, train-train
 

	model = GPy.models.GPRegression(x[train_low:train_up], y[train_low:train_up], kernel=kernel)
	model.optimize()
	y_pred = model.predict(x[train_up:])[0]
	y_true = y[train_up:]

	for cwgp in cwgp_model[::-1]:
		y_pred = cwgp.phi.inv_comp_phi(cwgp.phi.res.x, y_pred)
		y_true = cwgp.phi.inv_comp_phi(cwgp.phi.res.x, y_true)

	mae = mean_absolute_error(y_true, y_pred)
	rmse = mean_squared_error(y_true, y_pred, squared=False)

	print(mae,rmse)
	return model, mae, rmse


	

def grid_search(estimator, x, y, params):
	c = params.pop("c",2)
	n = params.pop("n",[2,3])
	transformations = params.pop("transformations",["sa","sal","box_cox"])

	cwgp_params = [transformations,n]

	params_product =  list(itertools.product(*cwgp_params))
	params_combination = []

	for param in itertools.permutations(params_product,c):
		if param not in params_combination:
			params_combination.append(param)

	cwgp = {}
	for index,param in enumerate(params_combination):
		t_data = copy.deepcopy(y)
		cwgp[index] = {"cwgp_combination":param}
		model_holder = []
		for t,d in param:
			cwgp_model = CWGP(t,n=d)
			cwgp_model.fit(t_data)
			t_data, t_data_d = cwgp_model.phi.comp_phi(cwgp_model.phi.res.x,t_data)
			model_holder.append(cwgp_model)
		cwgp[index]["result"] = estimator(x,t_data,model_holder)
	return cwgp

if __name__ == '__main__':
	CSV_FIlE = "../japan3.csv"
	df = pd.read_csv(CSV_FIlE)
	age = 90
	df_all = {}
	df_all[age]= df[(df["age"]==age)]
	
	rate = df_all[age]["rate"].to_numpy().reshape(-1,1)
	year = df_all[age]["year"].to_numpy().reshape(-1,1)

	result = grid_search(gp_regression, year, rate,{})
	print(result)
