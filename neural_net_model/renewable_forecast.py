import model_classes, nets

import torch
from torch.autograd import Variable, Function
import torch.cuda

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from datetime import datetime as dt

# Note: this assumes you're running it in the parent directory 
WEATHER_TRAIN = os.path.join('data', 'weather_formatting', 'formatted', 'weather_train_after_pca.csv')
WEATHER_TEST = os.path.join('data', 'weather_formatting', 'formatted', 'weather_test_after_pca.csv')
GENERATION_DATA = os.path.join('data', 'generation', 'generation_data.csv')
RENEWABLE_GEN = 'renewable_gen'
NUCLEAR_COL_NAME = 'nuclear_gen_prev_week'
LOAD_FORECAST_COL_NAME = 'pjm_load_forecast'
FORECASTED_FOSSIL_COL_NAME = 'fossil_gen'

INDEX = ['year', 'month', 'day']
DATE_COL = 'date'
ACTUAL_FOSSIL_GEN_2017 = os.path.join(os.path.dirname(os.getcwd()), 'data', 'generation', 'actual_fossil_gen_2017.csv')
FOSSIL_FORECAST_COL = ['date', 'forecasted fossil demand']

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

	model_rmse = model_classes.Net(X_Train, y_Train, [200, 200])
	model_rmse.cuda()

	model_rmse, iteration_list, train_loss_arr, test_loss_arr = nets.run_rmse_net(model_rmse, variables_rmse, X_Train, y_Train)

	train_rmse, test_rmse, pred_train, pred_test = nets.eval_net(model_rmse, variables_rmse)

	pred_values = pred_test.cpu().detach().numpy()
	pred_values = pd.DataFrame(pred_values, index = test_dates)
	datetime_fossil_demand(pred_values)


def datetime_fossil_demand(df):
	df, dates = merge_datetime_col(df, RENEWABLE_GEN)

	nuclear_df = get_nuclear(dates)
	df = df.reset_index().merge(nuclear_df, how='inner', on=DATE_COL)

	load_forecasts_df = get_load_forecasts()
	df = df.merge(load_forecasts_df, how='inner', on=DATE_COL)

	df[FORECASTED_FOSSIL_COL_NAME] = df[LOAD_FORECAST_COL_NAME] - df[RENEWABLE_GEN] - df[NUCLEAR_COL_NAME]
	forecasted_fossil_df = df[[DATE_COL, FORECASTED_FOSSIL_COL_NAME]]

	# merge with actual demand before sept 
	actual_fossil_gen = pd.read_csv(ACTUAL_FOSSIL_GEN_2017, parse_dates = ['date'])
	
	forecasted_fossil_df.set_index(DATE_COL, inplace=True)
	actual_fossil_gen.set_index(DATE_COL, inplace=True)

	forecasted_fossil_df = forecasted_fossil_df.combine_first(actual_fossil_gen)
	forecasted_fossil_df.to_csv('forecasted_fossil_gen.csv')

	return forecasted_fossil_df



def get_nuclear(dates):
	df = pd.read_csv(GENERATION_DATA, index_col = INDEX)

	filter_col = [col for col in df if col.startswith(NUCLEAR_COL_NAME)]
	df = df[filter_col]

	df = df.reset_index().merge(dates,  on=['year', 'month', 'day'])

	df[DATE_COL] = pd.to_datetime(df.year * 1000 + df.day, format='%Y%j') + pd.to_timedelta(df.hour, unit='h')
	df = df.drop(['year', 'month', 'day', 'hour'], axis = 1)

	return df


def get_load_forecasts():
	df = pd.read_csv(GENERATION_DATA, index_col = INDEX)

	filter_col = [col for col in df if col.startswith(LOAD_FORECAST_COL_NAME)]
	df = df[filter_col]
	df, dates = merge_datetime_col(df, LOAD_FORECAST_COL_NAME)

	return df


def merge_datetime_col(df, col_name):
	df.reset_index(inplace=True)

	df = pd.melt(df, id_vars=['year', 'month', 'day'], var_name='hour', value_name=col_name)

	if col_name != RENEWABLE_GEN: 
		df.hour = df.hour.str.split('|').str[1]
	
	df.hour = df.hour.astype(int)

	dates = df[['year', 'month', 'day', 'hour']]
	df[DATE_COL] = pd.to_datetime(df.year * 1000 + df.day, format='%Y%j') + pd.to_timedelta(df.hour, unit='h')

	df = df.drop(['year', 'month', 'day', 'hour'], axis = 1)
	return df, dates


def load_data():
	X_train_df = pd.read_csv(WEATHER_TRAIN, index_col = INDEX)
	X_test_df = pd.read_csv(WEATHER_TEST, index_col = INDEX)
	target_df = pd.read_csv(GENERATION_DATA, index_col = INDEX)

	filter_col = [col for col in target_df if col.startswith(RENEWABLE_GEN)]
	target_df = target_df[filter_col]

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

	return X_train_df, X_test_df, y_train_df, y_test_df, test_dates


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


if __name__=='__main__':
    main()
    