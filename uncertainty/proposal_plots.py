import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from matplotlib import rc
import os
import numpy as np
import matplotlib.pyplot as plt

MEF_FORECASTS = os.path.join(os.path.dirname(os.getcwd()), 'mef_forecast_results.csv')

FORECASTED_FOSSIL_GEN = os.path.join(os.path.dirname(os.getcwd()), 'neural_net_model', 'forecasted_fossil_gen.csv')
ACTUAL_FOSSIL_GEN = os.path.join(os.path.dirname(os.getcwd()), 'data', 'generation', 'actual_fossil_gen_2017.csv')

ACTUAL_FOSSIL_GEN_NAME = 'baseline_fossil_demand'
FORECASTED_FOSSIL_GEN_NAME = 'forecasted_fossil_demand'
FOSSIL_GEN_DIFFERENCE = 'fossil_gen_difference'
DATE_COL = 'date'

WEATHER_WITH_NOISE = os.path.join(os.path.dirname(os.getcwd()), 'data', 'weather_formatting', 'formatted', 'weather_test_with_noise.csv')
WEATHER_ACTUAL = os.path.join(os.path.dirname(os.getcwd()), 'data', 'weather_formatting', 'formatted', 'weather_test.csv')
GENERATION_DATA = os.path.join(os.path.dirname(os.getcwd()), 'data', 'generation', 'generation_data.csv')
DATE_INDEX = ['year', 'month', 'day']
NUCLEAR_ACTUAL = 'nuclear_gen'
NUCLEAR_PREV_WEEK = 'nuclear_gen_prev_week'

mef_results = pd.read_csv(MEF_FORECASTS, index_col = 'date', parse_dates = [1], usecols=[1, 2, 3, 4]).dropna()
actual_fossil_gen = pd.read_csv(ACTUAL_FOSSIL_GEN, index_col = DATE_COL, parse_dates = [DATE_COL])
actual_fossil_gen.columns = [ACTUAL_FOSSIL_GEN_NAME]

forecasted_fossil_gen = pd.read_csv(FORECASTED_FOSSIL_GEN, index_col = DATE_COL, parse_dates = [DATE_COL])
forecasted_fossil_gen.columns = [FORECASTED_FOSSIL_GEN_NAME]

compare_fossil_gen = pd.merge(actual_fossil_gen, forecasted_fossil_gen, left_index=True, right_index=True)
compare_fossil_gen[FOSSIL_GEN_DIFFERENCE] = (compare_fossil_gen[ACTUAL_FOSSIL_GEN_NAME] - compare_fossil_gen[FORECASTED_FOSSIL_GEN_NAME]) ** 2

compare_fossil_gen = compare_fossil_gen[compare_fossil_gen[FOSSIL_GEN_DIFFERENCE] != 0]
compare_fossil_gen = pd.merge(mef_results, compare_fossil_gen, left_index=True, right_index=True) 
compare_fossil_gen["ground truth"] = pd.to_numeric(compare_fossil_gen["ground truth"])
date = compare_fossil_gen.loc['2017-09-20']

date['hour'] = date.index.hour
date.set_index('hour', inplace=True)


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)



# Example data
# t = np.arange(0.0, 1.0 + 0.01, 0.01)
# s = np.cos(4 * np.pi * t) + 2

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.plot(t, s)

# plt.xlabel(r'\textbf{time} (s)')
# plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
# plt.title(r"\TeX\ is Number "
#           r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
#           fontsize=16, color='gray')
# # Make room for the ridiculously large title.
# plt.subplots_adjust(top=0.8)

# plt.savefig('tex_demo')
# plt.show()


# My plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig,ax = plt.subplots()

plt.title(r"MEF Forecasts for 09/20/17", fontsize=14)
plt.plot(date['ground truth'], label=r'Proxy ground truth')
plt.plot(date['neural net'], label=r'Neural net baseline')
plt.plot(date['simple dispatch'], label=r'Dispatch with forecasted inputs baseline')
#plt.plot(date['ground truth'])
# plt.plot(date['neural net'])
# plt.plot(date['simple dispatch'])

#plt.ylabel('Marginal CO$_\mathrm{2}$ Emissions [kg/MWh]')
#plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
           # hspace = 0, wspace = 0)
plt.margins(0,0)
plt.ylabel(r'Marginal ' 
	r'$CO_2$'
	r' Emissions Factor (kg/MWh)')
plt.xlabel(r'Hour')

plt.legend(loc = 'lower center' , bbox_to_anchor=(0.5, -0.35), fontsize='small')
fig.subplots_adjust(bottom=0.23)
plt.savefig('mef_plot.pgf')
plt.show()

