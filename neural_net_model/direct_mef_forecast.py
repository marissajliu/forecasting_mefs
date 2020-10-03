import model_classes, nets

import torch
from torch.autograd import Variable, Function
import torch.cuda

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from datetime import datetime as dt

# Features
WEATHER_TRAIN = os.path.join(os.path.dirname(os.getcwd()), 'data', 'weather_formatting', 'formatted', 'weather_train_after_pca.csv')
WEATHER_TEST = os.path.join(os.path.dirname(os.getcwd()), 'data', 'weather_formatting', 'formatted', 'weather_test_after_pca.csv')

GENERATION_DATA = os.path.join(os.path.dirname(os.getcwd()), 'data', 'generation', 'generation_data.csv')
NUCLEAR_COL_NAME = 'nuclear_gen_prev_week'
LOAD_FORECAST_COL_NAME = 'pjm_load_forecast'

INDEX = ['year', 'month', 'day']
DATE_COL = 'date'

# Target 
BASELINE_MEF_2016 = os.path.join(os.path.dirname(os.getcwd()), 'simple_dispatch_model', 'simulated', 'final_mefs_PJM_2016_actual.csv')
BASELINE_MEF_2017 = os.path.join(os.path.dirname(os.getcwd()), 'simple_dispatch_model', 'simulated', 'final_mefs_PJM_2017_actual.csv')
USECOLS_MEFS = [1, 5]


def main():
	X_Train, X_Test, y_Train, y_Test, test_dates = load_data()

	X_Train = X_Train.values
	X_Test = X_Test.values
	y_Train = y_Train.values
	y_Test = y_Test.values

	# Construct tensors
	X_train_ = torch.from_numpy(X_Train).float().cuda()
	Y_train_ = torch.from_numpy(y_Train).float().cuda()
	X_test_ = torch.from_numpy(X_Test).float().cuda()
	Y_test_ = torch.from_numpy(y_Test).float().cuda()

	variables_rmse = {'X_train_': X_train_, 'Y_train_': Y_train_, 'X_test_': X_test_, 'Y_test_': Y_test_}

	model_rmse = model_classes.Net(X_Train, y_Train, [400, 400])
	model_rmse.cuda()

	model_rmse, iteration_list, train_loss_arr, test_loss_arr = nets.run_rmse_net(model_rmse, variables_rmse, X_Train, y_Train)

	train_rmse, test_rmse, pred_train, pred_test = nets.eval_net(model_rmse, variables_rmse)

	pred_values = pred_test.cpu().detach().numpy()
	pred_values = pd.DataFrame(pred_values, index = test_dates)
	format_forecasts(pred_values).to_csv('direct_mef_forecast.csv')


def format_forecasts(df):
	df = pd.melt(df, id_vars=['year', 'month', 'day'], var_name='hour', value_name='direct_mef_forecast')
	df.hour = df.hour.astype(int)

	df[DATE_COL] = pd.to_datetime(df.year * 1000 + df.day, format='%Y%j') + pd.to_timedelta(df.hour, unit='h')
	df = df.drop(['year', 'month', 'day', 'hour'], axis = 1)
	return df.set_index(DATE_COL)

def load_data():
	X_train_df = pd.read_csv(WEATHER_TRAIN, index_col = INDEX)
	X_test_df = pd.read_csv(WEATHER_TEST, index_col = INDEX)
	gen_data = pd.read_csv(GENERATION_DATA, index_col = INDEX)

	# get target data
	target_df = _get_baseline_mef_data()

	# Get relevant generation data columns, add to dataframe 
	filter_col = [col for col in target_df if col.startswith((NUCLEAR_COL_NAME, LOAD_FORECAST_COL_NAME))]
	gen_data = gen_data[filter_col]
	X_train_df = pd.merge(X_train_df, gen_data, left_index=True, right_index=True)
	X_test_df = pd.merge(X_test_df, gen_data, left_index=True, right_index=True)
	X_train_df.dropna(inplace=True)
	X_test_df.dropna(inplace=True)

	# Split target 
	y_train_df = target_df.drop(target_df.index.difference(X_train_df.index))
	y_test_df = target_df.drop(target_df.index.difference(X_test_df.index))

	X_train_df= X_train_df.drop(X_train_df.index.difference(y_train_df.index))
	X_test_df= X_test_df.drop(X_test_df.index.difference(y_test_df.index))

	test_dates = y_test_df.index

	# sin / cos correction 
	X_train_df = sin_cos_correction(X_train_df)
	X_test_df = sin_cos_correction(X_test_df)

	# standardize data 
	X_train_df, X_test_df = standardize_data(X_train_df, X_test_df)

	# Save the target for the similarity score calc 
	y_test_df.to_csv('direct_mef_forecast_target.csv')

	return X_train_df, X_test_df, y_train_df, y_test_df, test_dates


def _get_baseline_mef_data():
	baseline_2016 = pd.read_csv(BASELINE_MEF_2016, usecols = USECOLS_MEFS, parse_dates=['datetime'])
	baseline_2017 = pd.read_csv(BASELINE_MEF_2017, usecols = USECOLS_MEFS, parse_dates=['datetime'])
	co2_marg = pd.concat([baseline_2016, baseline_2017], ignore_index=True)

	# Remove outliers (defined as values < 2000, there are ~50 values)
	co2_marg = co2_marg[co2_marg.co2_marg < 2000]

	# Separate datetime col 
	co2_marg = _extract_date(co2_marg, 'datetime')
	co2_marg.drop('datetime', inplace=True, axis = 1)

	# Pivot 
	co2_marg = co2_marg.pivot_table(index = ['year','month', 'day'], columns="hour", values=['co2_marg'])
	co2_marg.columns = co2_marg.columns.map('{0[0]}|{0[1]}'.format)

	# 29 days are dropped 
	co2_marg.dropna(inplace = True)

	return co2_marg


def sin_cos_correction(df):
    df = df.reset_index()
    
    df['day_sin'] = np.sin((df['day']-1)*(2.*np.pi/365))
    df['day_cos'] = np.cos((df['day']-1)*(2.*np.pi/365))
    df['month_sin'] = np.sin((df['month']-1)*(2.*np.pi/12))
    df['month_cos'] = np.cos((df['month']-1)*(2.*np.pi/12))
    
    df = df.drop(['month', 'day'], axis=1)
    df = df.set_index(['year', 'month_sin', 'month_cos', 'day_sin', 'day_cos'])
    return df


def standardize_data(X_train, X_test):
	# Standardize features 
	scaler = StandardScaler()
	scaler.fit(X_train.values) # fit on training set 

	scaled_X_train = scaler.transform(X_train.values)
	scaled_X_test = scaler.transform(X_test.values)

	# Convert back to df after scaling 
	X_train = pd.DataFrame(scaled_X_train, index=X_train.index, columns=X_train.columns)
	X_test = pd.DataFrame(scaled_X_test, index=X_test.index, columns=X_test.columns)

	return X_train, X_test

def _extract_date(df, date_col):
    df.copy(deep=True)
    df['hour'] = df[date_col].dt.hour
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['day'] = df[date_col].dt.dayofyear
    return df 

if __name__=='__main__':
    main()
