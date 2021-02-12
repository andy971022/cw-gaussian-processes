import itertools
from cwgp.grid_search import grid_search
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import GPy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os




def cwgp_regression(x, y, cwgp_model):
	# kernel = GPy.kern.RBF(1)
	kernel = GPy.kern.Matern32(1)

	ratio = 0.9
	length = len(rate)
	train = int(ratio*length)
	train_up, train_low = train, train-train

	avg = np.mean(y[train_low:train_up])

	mf = GPy.core.Mapping(1,1)
	mf.f = lambda x: 0
	mf.update_gradients = lambda a,b: None


	model = GPy.models.GPRegression(x[train_low:train_up], y[train_low:train_up], mean_function=mf, kernel=kernel)
	model.optimize()

	domain_discrete = np.arange(1940,2031).reshape(-1,1)
	y_res = model.predict(x)
	y_all = model.predict(domain_discrete)
	y_pred= y_res[0]
	y_mean,y_var = y_all[0],y_all[1]
	y_top, y_bot = y_mean + 1.96*np.sqrt(y_var), y_mean - 1.96*np.sqrt(y_var)

	for cwgp in cwgp_model[::-1]:
		y_pred = cwgp.phi.inv_comp_phi(cwgp.phi.res.x, y_pred)
		y_mean, y_var = cwgp.phi.inv_comp_phi(cwgp.phi.res.x, y_mean), cwgp.phi.inv_comp_phi(cwgp.phi.res.x, y_var)
		y_top, y_bot = cwgp.phi.inv_comp_phi(cwgp.phi.res.x, y_top), cwgp.phi.inv_comp_phi(cwgp.phi.res.x, y_bot)
		y = cwgp.phi.inv_comp_phi(cwgp.phi.res.x, y)


	print(y_pred[train_up:])
	rmse = mean_squared_error(y[train_up:], y_pred[train_up:], squared=False)
	mae = mean_absolute_error(y[train_up:], y_pred[train_up:])
	txt = f"""rmse : {rmse} \n mae : {mae}"""

	name = [f"{cwgp.phi.fn.__name__}_{cwgp.phi.n}" for cwgp in cwgp_model]
	name = "_".join(name)
	if rmse < 0.001271134608025886:
		domain = np.linspace(1940,2030,91)
		plt.fill_between(domain, np.ravel(y_top), np.ravel(y_bot), color=(0,0.5,0.5,0.2), label="Confidence")
		plt.scatter(x[train_low:train_up], y[train_low:train_up], marker="x", color='black', label="data")
		plt.scatter(x[train_up:], y[train_up:], marker="x", color='red')
		plt.plot(np.linspace(1940,2030,91),y_mean, label="mean")
		plt.ylim([0,0.12])
		# plt.ylim([0.1,0.35])
		plt.legend()
		plt.figtext(0.5, 0.2, txt, wrap=True, horizontalalignment='center', fontsize=12)
		plt.grid(True)
		plt.savefig(f'./{IMG_DIR}/{name}_{age}_gs.png')
		plt.clf()
		

	print(rmse,mae)
	return rmse, mae



if __name__ == '__main__':
	CSV_FIlE = "../japan3.csv"
	IMG_DIR = "./gs"
	df = pd.read_csv(CSV_FIlE)
	age = 75
	df_all = {}
	df_all[age]= df[(df["age"]==age)]
	
	year = df_all[age]["year"].to_numpy().reshape(-1,1)
	rate = df_all[age]["rate"].to_numpy().reshape(-1,1)



	if not os.path.exists(IMG_DIR):
		os.makedirs(IMG_DIR)


	result = grid_search(cwgp_regression, year, rate, {"c":4,"n":[1,2],"transformations":["sa","sal","box_cox"]})


	output = pd.DataFrame(result).transpose()

	output.to_csv(f"./ssb_3_23_{age}.csv")