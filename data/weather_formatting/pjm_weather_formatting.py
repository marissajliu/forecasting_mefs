import pandas as pd
import numpy as np
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy 

'''
	Get formatted hourly PJM weather data
		(air temp, drew point temp, sky ceiling height, wind speed, sea level pressure)
	Input:  
		Data source: NOAA global hourly surface data
		Downloaded from: Dolthub (https://www.dolthub.com/blog/2020-03-02-noaa-global-hourly-surface-data/)
	Output: 
		Training and test set of:
			- weather with noise
			- weather without noise
			- weather with noise after PCA 
		Saved to /formatted 
'''

pd.options.mode.chained_assignment = None

# DEFINE COLUMN LABELS 
WEATHER_TYPE = ['air_temp', 'wind_speed', 'sea_level_pressure', 'sky_ceiling_height', 'dew_point_temp']
YEAR_RANGE = [2016, 2017]
NUM_DAYS_IN_TEST = 122
MAX_MISSING_DAYS = 10
PCA_COMPONENTS = {'air_temp': 3, 'wind_speed': 25, 'sea_level_pressure': 25, 'sky_ceiling_height': 25, 'dew_point_temp': 3}


def main():
	all_weather_df_train = pd.DataFrame()
	all_weather_df_test = pd.DataFrame()
	all_weather_df_train_with_noise = pd.DataFrame()
	all_weather_df_test_with_noise = pd.DataFrame()
	all_weather_train_after_pca = pd.DataFrame()
	all_weather_test_after_pca = pd.DataFrame()


	for weather_type in WEATHER_TYPE:
		print("Current weather type: " + weather_type)
		weather_type_df = pd.DataFrame() 

		for year in YEAR_RANGE:
			print("Reading year " + str(year))
			#TODO: create a function called clean_yearly_data
			df = format_hourly_df(year, weather_type)

			# Handle missing values 
			by_day_df, df = drop_missing_data(df)
			days_filled = fill_missing_days(by_day_df, df)

			# combine data from the two years 
			weather_type_df = pd.concat([weather_type_df, days_filled])

		# pivot by hour 
		weather_type_df = weather_type_df.pivot_table(index = ['year','month', 'day'], columns="hour", values=weather_type_df.columns)
		weather_type_df.columns = weather_type_df.columns.map('{0[0]}|{0[1]}'.format)	

		weather_type_df.dropna(inplace=True, axis = 1)	
		weather_type_df.to_csv('interim.csv')

		# Split 
		weather_train, weather_test = train_test_split(weather_type_df, NUM_DAYS_IN_TEST)
		all_weather_df_train = pd.concat([all_weather_df_train, weather_train], axis = 1)
		all_weather_df_test = pd.concat([all_weather_df_test, weather_test], axis = 1)

		# Add noise and apply PCA 
		all_weather_df_train_with_noise = pd.concat([all_weather_df_train_with_noise, add_noise(weather_train)], axis = 1)
		all_weather_df_test_with_noise = pd.concat([all_weather_df_test_with_noise, add_noise(weather_test)], axis = 1)

		# apply pca to df with noise 
		weather_train_after_pca, weather_test_after_pca = apply_pca(weather_train, weather_test, PCA_COMPONENTS[weather_type])
		all_weather_train_after_pca = pd.concat([all_weather_train_after_pca, weather_train_after_pca], axis = 1)
		all_weather_test_after_pca = pd.concat([all_weather_test_after_pca, weather_test_after_pca], axis = 1)

	# save data 
	all_weather_df_train.to_csv('formatted/weather_train.csv')
	all_weather_df_test.to_csv('formatted/weather_test.csv')

	all_weather_df_train_with_noise.to_csv('formatted/weather_train_with_noise.csv')
	all_weather_df_test_with_noise.to_csv('formatted/weather_test_with_noise.csv')

	all_weather_train_after_pca.to_csv('formatted/weather_train_after_pca.csv')
	all_weather_test_after_pca.to_csv('formatted/weather_test_after_pca.csv')


def format_hourly_df(year, weather_type):
	'''
	Pivots data so there's a column per weather station and gets time attributes 

	Returns 
		df with year, month, day attribute and a column per weather station 
	''' 
	df = pd.read_csv("raw_data/" + weather_type + "/" + str(year) + ".csv", usecols=[1, 2, 3, 4], low_memory=False)

	# Handle bug where some hour columns contain the text "commit_date"
	df = df.loc[df['hour'] != 'commit_date']

	# Pivot so each weather station is a separate column
	df = df.set_index(['station','day', 'hour']).unstack('station')

	# Merge multiindex 
	df.columns = df.columns.map('{0[0]}|{0[1]}'.format)

	# Separate date column into year, month, day 
	df.reset_index(inplace=True)
	df['year'] = pd.to_datetime(df['day']).dt.year
	df['month'] = pd.to_datetime(df['day']).dt.month
	df['day'] = pd.to_datetime(df['day']).dt.dayofyear

	# Make all columns numeric 
	cols = df.columns
	df[cols] = df[cols].apply(pd.to_numeric)

	return df 


def drop_missing_data(df):
	''' Drop stations with more than MAX_MISSING_DAYS worth of missing data''' 

	# Find stations with > specified days of missing data
	by_day_df = df.copy(deep=True).groupby(['year', 'month', 'day']).mean()
	by_day_df.reset_index(inplace=True)
	by_day_df.dropna(thresh=len(by_day_df) - MAX_MISSING_DAYS, axis=1, inplace = True)

	# keep only columns with > specified days of data
	df = df[df.columns.intersection(by_day_df.columns)]
	by_day_df.drop('hour', axis = 1, inplace=True)

	df.set_index(['year', 'month', 'day', 'hour'], inplace=True)
	by_day_df.set_index(['year', 'month', 'day'], inplace=True)

	return by_day_df, df 


def fill_missing_days(by_day_df, df):
	# Create a df where True = there is no data for the entire day
		# False = there is data for at least 1 hour of the day 
	test = by_day_df.copy(deep=True)
	test = test.isnull()

	# Create a df with just year, month, day, and hour 
	time_col = pd.DataFrame(index=df.index)

	fill_missing_days = (by_day_df.ffill()+by_day_df.bfill())/2
	fill_missing_days = fill_missing_days.bfill().ffill()

	# Take the daily data and copy it to 24 hours
	avg_day_with_hourly_data = pd.merge(fill_missing_days, time_col, left_index=True, right_index=True)

	avg_day_with_hourly_data[test==False] = 99999


	df.fillna(avg_day_with_hourly_data, inplace=True)
	df.replace(99999, np.nan, inplace=True)

	# Fill missing hours 
	df = (df.ffill()+df.bfill())/2
	df = df.bfill().ffill()

	return df


def train_test_split(df, num_days):
	df = df.copy(deep=True)

	X_train = df.head(len(df.index) - num_days)
	X_test = df.tail(num_days)

	return X_train, X_test


def apply_pca(X_train, X_test, num_components):
	X_train = X_train.copy(deep=True)
	X_test = X_test.copy(deep=True)

	pca = PCA(n_components = num_components)
	pca.fit(X_train.values)

	X_train_pca = pca.transform(X_train.values)
	X_test_pca = pca.transform(X_test.values)

	# Convert back to df after pca 
	X_train = pd.DataFrame(X_train_pca, index=X_train.index)
	X_test = pd.DataFrame(X_test_pca, index=X_test.index)

	print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_.cumsum()))

	return X_train, X_test 


def add_noise(df):
	''' Add noise by finding the average of the difference between the weather values on two consecutive days, 
		then add normally distributed random noise that has the standard deviation equal to the previously mentioned average'''
	
	df = df.copy(deep=True)
	noise_std_dev = np.abs(df.diff()).mean(axis = 0)

	for col, val in noise_std_dev.items():
		noise = np.random.normal(0, noise_std_dev[col], len(df.index))
		df[col] = df[col] + noise

	return df


if __name__=='__main__':
    main()
