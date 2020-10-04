import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 


MEF_FORECASTS = os.path.join(os.path.dirname(os.getcwd()), 'mef_forecast_results.csv')

# FORECASTED_FOSSIL_GEN = os.path.join(os.path.dirname(os.getcwd()), 'neural_net_model', 'forecasted_fossil_gen.csv')
# ACTUAL_FOSSIL_GEN = os.path.join(os.path.dirname(os.getcwd()), 'data', 'generation', 'actual_fossil_gen_2017.csv')

# ACTUAL_FOSSIL_GEN_NAME = 'baseline_fossil_demand'
# FORECASTED_FOSSIL_GEN_NAME = 'forecasted_fossil_demand'
# SIMPLE_DISPATCH_SIMILARITY_SCORE = 'simple_dispatch_similarity_score'
# DATE_COL = 'date'

# WEATHER_WITH_NOISE = os.path.join(os.path.dirname(os.getcwd()), 'data', 'weather_formatting', 'formatted', 'weather_test_with_noise.csv')
# WEATHER_ACTUAL = os.path.join(os.path.dirname(os.getcwd()), 'data', 'weather_formatting', 'formatted', 'weather_test.csv')
# GENERATION_DATA = os.path.join(os.path.dirname(os.getcwd()), 'data', 'generation', 'generation_data.csv')
# DATE_INDEX = ['year', 'month', 'day']
# NUCLEAR_ACTUAL = 'nuclear_gen'
# NUCLEAR_PREV_WEEK = 'nuclear_gen_prev_week'

def main():
	calc_rmse()


def calc_rmse():
	mef_results = pd.read_csv(MEF_FORECASTS, index_col = 1, parse_dates = [1]).dropna()
	mef_results['baseline'] = mef_results["ground truth"].mean()
	print('average mef value is ', mef_results['baseline'][0])

	print('RMSE (Simple Dispatch): ', mean_squared_error(mef_results['ground truth'], mef_results['simple dispatch'], squared = False))
	print('RMSE (Neural Net): ', mean_squared_error(mef_results['ground truth'], mef_results['neural net'], squared = False))
	print('RMSE (Baseline): ', mean_squared_error(mef_results['ground truth'], mef_results['baseline'], squared = False))


# def simple_dispatch_similarity_score():
# 	''' Squared error of baseline fossil gen vs predicted fossil gen '''
# 	mef_results = pd.read_csv(MEF_FORECASTS, index_col = 1, parse_dates = [1]).dropna()
# 	actual_fossil_gen = pd.read_csv(ACTUAL_FOSSIL_GEN, index_col = DATE_COL, parse_dates = [DATE_COL])
# 	actual_fossil_gen.columns = [ACTUAL_FOSSIL_GEN_NAME]

# 	forecasted_fossil_gen = pd.read_csv(FORECASTED_FOSSIL_GEN, index_col = DATE_COL, parse_dates = [DATE_COL])
# 	forecasted_fossil_gen.columns = [FORECASTED_FOSSIL_GEN_NAME]

# 	compare_fossil_gen = pd.merge(actual_fossil_gen, forecasted_fossil_gen, left_index=True, right_index=True)
# 	compare_fossil_gen[SIMPLE_DISPATCH_SIMILARITY_SCORE] = (compare_fossil_gen[ACTUAL_FOSSIL_GEN_NAME] - compare_fossil_gen[FORECASTED_FOSSIL_GEN_NAME]) ** 2


# 	compare_fossil_gen = compare_fossil_gen[compare_fossil_gen[SIMPLE_DISPATCH_SIMILARITY_SCORE] != 0]
# 	compare_fossil_gen = pd.merge(compare_fossil_gen, mef_results, left_index=True, right_index=True)
# 	compare_fossil_gen = compare_fossil_gen.drop([FORECASTED_FOSSIL_GEN_NAME, FORECASTED_FOSSIL_GEN_NAME, 'neural net', 'Unnamed: 0'], axis = 1)

# 	compare_fossil_gen.to_csv('simple_dispatch_similarity_score.csv')
# 	return compare_fossil_gen

def filter_by_simple_dispatch_similarity_score(df, percentile):
	''' Return all values in the specified percentile (0.1 = top 10 percentile) '''
	value = df[SIMPLE_DISPATCH_SIMILARITY_SCORE].quantile(percentile)
	df = df[df[SIMPLE_DISPATCH_SIMILARITY_SCORE] <= value]
	df.to_csv('simple_dispatch_high_similarity_score.csv')


def neural_net_similarity_score():
	''' 
	Comparing:
	- Weather with noise, PJM fossil forecast, last week's nuclear gen 
	- Weather, PJM acutal fossil, current nuclear gen 
	Currently not accounting for difference between load forecasts 
	'''
	print('something')


if __name__=='__main__':
    main()
