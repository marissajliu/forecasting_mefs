import os
import pandas as pd
import numpy as np

BASELINE_MEFS = os.path.join('simple_dispatch_model', 'simulated', 'final_mefs_PJM_2017_actual.csv')
DIRECT_FORECAST = os.path.join('neural_net_model', 'direct_mef_forecast.csv')
SIMPLE_DISPATCH_FORECAST = os.path.join('simple_dispatch_model', 'simulated', 'final_mefs_PJM_2017_forecast.csv')

DATE_COL = 'date'
BASELINE_MEFS_COL = [DATE_COL, 'baseline']
DIRECT_MEFS_COL = [DATE_COL, 'neural net']
SIMPLE_DISPATCH_COL = [DATE_COL, 'simple dispatch']


def main():
	results = pd.DataFrame()
	baseline_df = pd.read_csv(BASELINE_MEFS, parse_dates = ['datetime'], usecols=[1, 5])
	direct_df = pd.read_csv(DIRECT_FORECAST, parse_dates = [DATE_COL])
	simple_dispatch_df = pd.read_csv(SIMPLE_DISPATCH_FORECAST, parse_dates = ['datetime'], usecols=[1, 5])
	
	baseline_df.columns = BASELINE_MEFS_COL
	direct_df.columns = DIRECT_MEFS_COL
	simple_dispatch_df.columns = SIMPLE_DISPATCH_COL

	_pivot_by_hour(baseline_df, 'baseline').to_csv('baseline_mef_by_hour.csv')
	_pivot_by_hour(direct_df, 'neural net').to_csv('neural_net_mef_by_hour.csv')
	_pivot_by_hour(simple_dispatch_df, 'simple dispatch').to_csv('simple_dispatch_mef_by_hour.csv')

	results = pd.merge(baseline_df, direct_df, on = DATE_COL, how = 'inner')
	results[DATE_COL] = pd.to_datetime(results[DATE_COL])

	results = pd.merge(results, simple_dispatch_df, on = DATE_COL, how = 'inner')
	results.to_csv('mef_forecast_results.csv')

	
def _pivot_by_hour(df, col_name):
	df = df.copy()
	df['hour'] = df[DATE_COL].dt.hour
	df['date'] = df[DATE_COL].dt.date

	df = df.pivot_table(index = ['date'], columns='hour', values=[col_name])
	df.columns = df.columns.map('{0[0]}|{0[1]}'.format)

	return df 

if __name__=='__main__':
    main()