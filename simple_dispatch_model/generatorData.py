import pandas
import matplotlib.pylab
import scipy
import scipy.interpolate
import datetime
import math
import copy
from bisect import bisect_left
import os 

# class "generatorData" turns CEMS, eGrid, FERC, and EIA data into a cleaned up dataframe for feeding into a "bidStack" object

class generatorData(object):
    def __init__(self, nerc, egrid_fname, eia923_fname, demand, ferc714IDs_fname='', ferc714_fname='', cems_folder='', easiur_fname='', include_easiur_damages=True, year=2015, fuel_commodity_prices_excel_dir='', hist_downtime = True, coal_min_downtime = 12, cems_validation_run=False):
        """
        Translates the CEMS, eGrid, FERC, and EIA data into a dataframe for feeding into the bidStack class
        ---
        nerc : nerc region of interest (e.g. 'TRE', 'MRO', etc.)
        egrid_fname : a .xlsx file name for the eGrid generator data
        eia923_fname : filename of eia form 923
        ferc714_fname : filename of nerc form 714 hourly system lambda
        ferc714IDs_fname : filename that matches nerc 714 respondent IDs with nerc regions
        easiur_fname : filename containing easiur damages ($/tonne) for each power plant orispl
        include_easiur_damages : if True, then easiur damages will be added to the generator data. If False, we will skip that step.
        year : year that we're looking at (e.g. 2017)
        fuel_commodity_prices_excel_dir : filename of national EIA fuel prices (in case EIA923 data is empty)
        hist_downtime : if True, will use each generator's maximum weekly mw for weekly capacity. if False, will use maximum annual mw.
        coal_min_downtime : hours that coal plants must remain off if they shut down
        cems_validation_run : if True, then we are trying to validate the output against CEMS data and will only look at power plants included in the CEMS data
        """

        #read in the data. This is a bit slow right now because it reads in more data than needed, but it is simple and straightforward
        self.nerc = nerc
        egrid_year_str = str(math.floor((year / 2.0)) * 2)[2:4] #eGrid is only every other year so we have to use eGrid 2016 to help with a 2017 run, for example

        print('Reading in unit level data from eGRID...')
        self.egrid_unt = pandas.read_excel(egrid_fname, 'UNT'+egrid_year_str, skiprows=[0])
        print ('Reading in generator level data from eGRID...')
        self.egrid_gen = pandas.read_excel(egrid_fname, 'GEN'+egrid_year_str, skiprows=[0])
        print ('Reading in plant level data from eGRID...')
        self.egrid_plnt = pandas.read_excel(egrid_fname, 'PLNT'+egrid_year_str, skiprows=[0])
        print ('Reading in data from EIA Form 923...')
        eia923 = pandas.read_excel(eia923_fname, 'Page 5 Fuel Receipts and Costs', skiprows=[0,1,2,3])
        eia923 = eia923.rename(columns={'Plant Id': 'orispl'})
        self.eia923 = eia923
        eia923_1 = pandas.read_excel(eia923_fname, 'Page 1 Generation and Fuel Data', skiprows=[0,1,2,3,4])
        eia923_1 = eia923_1.rename(columns={'Plant Id': 'orispl'})
        self.eia923_1 = eia923_1
        print ('Reading in data from FERC Form 714...')
        self.ferc714 = pandas.read_csv(ferc714_fname)
        self.ferc714_ids = pandas.read_csv(ferc714IDs_fname)
        self.cems_folder = cems_folder
        self.easiur_per_plant = pandas.read_csv(easiur_fname)
        self.fuel_commodity_prices = pandas.read_excel(fuel_commodity_prices_excel_dir, str(year))
        self.cems_validation_run = cems_validation_run
        self.hist_downtime = hist_downtime
        self.coal_min_downtime = coal_min_downtime
        self.year = year

        self.cleanGeneratorData(year)
        self.addGenVom()
        self.calcFuelPrices()
        if include_easiur_damages:
            self.easiurDamages()
        self.addGenMinOut()
        self.addDummies()
        self.calcDemandData(demand)
        #self.addElecPriceToDemandData()
        self.demandTimeSeries(demand)
        self.calcMdtCoalEvents()


    def cleanGeneratorData(self, year):
        """
        Converts the eGrid and CEMS data into a dataframe usable by the bidStack class.
        ---
        Creates
        self.df : has 1 row per generator unit or plant. columns describe emissions, heat rate, capacity, fuel, grid region, etc. This dataframe will be used to describe the generator fleet and merit order.
        self.df_cems : has 1 row per hour of the year per generator unit or plant. columns describe energy generated, emissions, and grid region. This dataframe will be used to describe the historical hourly demand, dispatch, and emissions
        """
        #copy in the egrid data and merge it together. In the next few lines we use the eGRID excel file to bring in unit level data for fuel consumption and emissions, generator level data for capacity and generation, and plant level data for fuel type and grid region. Then we compile it together to get an initial selection of data that defines each generator.
        print ('Cleaning eGRID Data...')
        #unit-level data
        df = self.egrid_unt.copy(deep=True)
        #rename columns
        df = df[['PNAME', 'ORISPL', 'UNITID', 'PRMVR', 'FUELU1', 'HTIAN', 'NOXAN', 'SO2AN', 'CO2AN', 'HRSOP']]
        df.columns = ['gen', 'orispl', 'unit', 'prime_mover', 'fuel', 'mmbtu_ann', 'nox_ann', 'so2_ann', 'co2_ann', 'hours_on']
        df['orispl_unit'] = df.orispl.map(str) + '_' + df.unit.map(str) #orispl_unit is a unique tag for each generator unit
        
        #drop nan fuel
        df = df[~df.fuel.isna()]
        
        #gen-level data: contains MW capacity and MWh annual generation data
        df_gen = self.egrid_gen.copy(deep=True)
        df_gen['orispl_unit'] = df_gen['ORISPL'].map(str) + '_' + df_gen['GENID'].map(str) #orispl_unit is a unique tag for each generator unit
        df_gen_long = df_gen[['ORISPL', 'NAMEPCAP', 'GENNTAN', 'GENYRONL', 'orispl_unit', 'PRMVR', 'FUELG1']].copy()
        df_gen_long.columns = ['orispl', 'mw', 'mwh_ann', 'year_online', 'orispl_unit', 'prime_mover', 'fuel']
        df_gen = df_gen[['NAMEPCAP', 'GENNTAN', 'GENYRONL', 'orispl_unit']]
        df_gen.columns = ['mw', 'mwh_ann', 'year_online', 'orispl_unit']
        #plant-level data: contains fuel, fuel_type, balancing authority, nerc region, and egrid subregion data
        df_plnt = self.egrid_plnt.copy(deep=True)
        #fuel
        df_plnt_fuel = df_plnt[['PLPRMFL', 'PLFUELCT']]
        df_plnt_fuel = df_plnt_fuel.drop_duplicates('PLPRMFL')
        df_plnt_fuel.PLFUELCT = df_plnt_fuel.PLFUELCT.str.lower()
        df_plnt_fuel.columns = ['fuel', 'fuel_type']
        
        #geography
        df_plnt = df_plnt[['ORISPL', 'BACODE', 'ISORTO', 'SUBRGN']]
        df_plnt.columns = ['orispl', 'ba', 'isorto', 'egrid']
        
        #merge these egrid data together at the unit-level
        df = df.merge(df_gen, left_index=True, how='left', on='orispl_unit')
        df = df.merge(df_plnt, left_index=True, how='left', on='orispl')
        df = df.merge(df_plnt_fuel, left_index=True, how='left', on='fuel')
        
        #keep only the units in the nerc region we're analyzing
        df = df[df.isorto == self.nerc]
        
        #calculate the emissions rates
        df['co2'] = scipy.divide(df.co2_ann,df.mwh_ann) * 907.185 #tons to kg
        df['so2'] = scipy.divide(df.so2_ann,df.mwh_ann) * 907.185 #tons to kg
        df['nox'] = scipy.divide(df.nox_ann,df.mwh_ann) * 907.185 #tons to kg
        
        #for empty years, look at orispl in egrid_gen instead of orispl_unit
        df.loc[df.year_online.isna(), 'year_online'] = df[df.year_online.isna()][['orispl', 'prime_mover', 'fuel']].merge(df_gen_long[['orispl', 'prime_mover', 'fuel', 'year_online']].groupby(['orispl', 'prime_mover', 'fuel'], as_index=False).agg('mean'), on=['orispl', 'prime_mover', 'fuel'])['year_online']
        #for any remaining empty years, assume self.year (i.e. that they are brand new)
        df.loc[df.year_online.isna(), 'year_online'] = scipy.zeros_like(df.loc[df.year_online.isna(), 'year_online']) + self.year
        ###
        #now sort through and compile CEMS data. The goal is to use CEMS data to characterize each generator unit. So if CEMS has enough information to describe a generator unit we will over-write the eGRID data. If not, we will use the eGRID data instead. (CEMS data is expected to be more accurate because it has actual hourly performance of the generator units that we can use to calculate their operational characteristics. eGRID is reported on an annual basis and might be averaged out in different ways than we would prefer.)
        print ('Compiling CEMS data...')
        #dictionary of which states are in which nerc region (b/c CEMS file downloads have the state in the filename)
        PJM_STATES = ['mi','il', 'in', 'ky', 'oh', 'wv', 'pa', 'md', 'de', 'nj', 'va', 'md', 'de']
        #compile the different months of CEMS files into one dataframe, df_cems. (CEMS data is downloaded by state and by month, so compiling a year of data for ERCOT / TRE, for example, requires reading in 12 Texas .csv files and 12 Oklahoma .csv files)
        df_cems = pandas.DataFrame()
        for s in PJM_STATES:
            for m in ['01','02','03','04','05','06','07','08','09','10','11', '12']:
                print (s + ': ' + m)
                df_cems_add = pandas.read_csv(self.cems_folder + str(year) + s + m + '.csv')
                df_cems_add = df_cems_add[['ORISPL_CODE', 'UNITID', 'OP_DATE','OP_HOUR','GLOAD (MW)', 'SO2_MASS (lbs)', 'NOX_MASS (lbs)', 'CO2_MASS (tons)', 'HEAT_INPUT (mmBtu)']].dropna()
                df_cems_add.columns=['orispl', 'unit', 'date','hour','mwh', 'so2_tot', 'nox_tot', 'co2_tot', 'mmbtu']
                df_cems = pandas.concat([df_cems, df_cems_add])
        
        #create the 'orispl_unit' column, which combines orispl and unit into a unique tag for each generation unit
        df_cems['orispl_unit'] = df_cems['orispl'].map(str) + '_' + df_cems['unit'].map(str)
        
        #bring in geography data and only keep generators within self.nerc
        df_cems = df_cems.merge(df_plnt, left_index=True, how='left', on='orispl')
        df_cems = df_cems[df_cems['isorto']==self.nerc]
        
        #convert emissions to kg
        df_cems.co2_tot = df_cems.co2_tot * 907.185 #tons to kg
        df_cems.so2_tot = df_cems.so2_tot * 0.454 #lbs to kg
        df_cems.nox_tot = df_cems.nox_tot * 0.454 #lbs to kg
        
        #calculate the hourly heat and emissions rates. Later we will take the medians over each week to define the generators weekly heat and emissions rates.
        df_cems['heat_rate'] = df_cems.mmbtu / df_cems.mwh
        df_cems['co2'] = df_cems.co2_tot / df_cems.mwh
        df_cems['so2'] = df_cems.so2_tot / df_cems.mwh
        df_cems['nox'] = df_cems.nox_tot / df_cems.mwh
        df_cems.replace([scipy.inf, -scipy.inf], scipy.nan, inplace=True) #don't want inf messing up median calculations
        
        #drop any bogus data. For example, the smallest mmbtu we would expect to see is 25MW(smallest unit) * 0.4(smallest minimum output) * 6.0 (smallest heat rate) = 60 mmbtu. Any entries with less than 60 mmbtu fuel or less than 6.0 heat rate, let's get rid of that row of data.
        df_cems = df_cems[(df_cems.heat_rate >= 6.0) & (df_cems.mmbtu >= 60)]
        
        #calculate emissions rates and heat rate for each week and each generator
        #rather than parsing the dates (which takes forever because this is such a big dataframe) we can create month and day columns for slicing the data based on time of year
        df_orispl_unit = df_cems.copy(deep=True)
        df_orispl_unit.date = df_orispl_unit.date.str.replace('/','-')
        temp = pandas.DataFrame(df_orispl_unit.date.str.split('-').tolist(), columns=['month', 'day', 'year'], index=df_orispl_unit.index).astype(float)
        df_orispl_unit['monthday'] = temp.year*10000 + temp.month*100 + temp.day
        ###
        #loop through the weeks, slice the data, and find the average heat rates and emissions rates
        #first, add a column 't' that says which week of the simulation we are in
        df_orispl_unit['t'] = 52
        for t in scipy.arange(52)+1:
            start = (datetime.datetime.strptime(str(self.year) + '-01-01', '%Y-%m-%d') + datetime.timedelta(days=7.05*(t-1)-1)).strftime('%Y-%m-%d')
            end = (datetime.datetime.strptime(str(self.year) + '-01-01', '%Y-%m-%d') + datetime.timedelta(days=7.05*(t)-1)).strftime('%Y-%m-%d')
            start_monthday = float(start[0:4])*10000 + float(start[5:7])*100 + float(start[8:])
            end_monthday = float(end[0:4])*10000 + float(end[5:7])*100 + float(end[8:])
            #slice the data for the days corresponding to the time series period, t
            df_orispl_unit.loc[(df_orispl_unit.monthday >= start_monthday) & (df_orispl_unit.monthday < end_monthday), 't'] = t
        #remove outlier emissions and heat rates. These happen at hours where a generator's output is very low (e.g. less than 10 MWh). To remove these, we will remove any datapoints where mwh < 10.0 and heat_rate < 30.0 (0.5% percentiles of the 2014 TRE data).
        df_orispl_unit = df_orispl_unit[(df_orispl_unit.mwh >= 10.0) & (df_orispl_unit.heat_rate <= 30.0)]
        #aggregate by orispl_unit and t to get the heat rate, emissions rates, and capacity for each unit at each t
        temp_2 = df_orispl_unit.groupby(['orispl_unit', 't'], as_index=False).agg('median')[['orispl_unit', 't', 'heat_rate', 'co2', 'so2', 'nox']].copy(deep=True)
        temp_2['mw'] = df_orispl_unit.groupby(['orispl_unit', 't'], as_index=False).agg('max')['mwh'].copy(deep=True)
        #condense df_orispl_unit down to where we just have 1 row for each unique orispl_unit
        df_orispl_unit = df_orispl_unit[['orispl_unit', 'orispl', 'ba', 'isorto', 'egrid', 'mwh']]

        df_orispl_unit = df_orispl_unit.groupby('orispl_unit', as_index=False).agg('max')
        df_orispl_unit.rename(columns={'mwh':'mw'}, inplace=True)
        for c in ['heat_rate', 'co2', 'so2', 'nox', 'mw']:
            temp_3 = temp_2.set_index(['orispl_unit', 't'])[c].unstack().reset_index()
            temp_3.columns = list(['orispl_unit']) + ([c + str(a) for a in scipy.arange(52)+1])
            if not self.hist_downtime:
                #remove any outlier values in the 1st or 99th percentiles
                max_array = temp_3.copy().drop(columns='orispl_unit').quantile(0.99, axis=1)
                min_array = temp_3.copy().drop(columns='orispl_unit').quantile(0.01, axis=1)
                median_array = temp_3.copy().drop(columns='orispl_unit').median(axis=1)
                for i in temp_3.index:
                    test = temp_3.drop(columns='orispl_unit').iloc[i]
                    test[test > max_array[i]] = scipy.NaN
                    test[test < min_array[i]] = scipy.NaN
                    test = list(test) #had a hard time putting the results back into temp_3 without using a list
                    #if the first entry in test is nan, we want to fill that with the median value so that we can use ffill later
                    if math.isnan(test[0]):
                        test[0] = median_array[i]
                    test.insert(0, temp_3.iloc[i].orispl_unit)
                    temp_3.iloc[i] = test
            #for any nan values (assuming these are offline generators without any output data), fill nans with a large heat_rate that will move the generator towards the end of the merit order and large-ish emissions rate, so if the generator is dispatched in the model it will jack up prices but emissions won't be heavily affected (note, previously I just replaced all nans with 99999, but I was concerned that this might lead to a few hours of the year with extremely high emissions numbers that threw off the data)
            M = float(scipy.where(c=='heat_rate', 50.0, scipy.where(c=='co2', 1500.0, scipy.where(c=='so2', 4.0, scipy.where(c=='nox', 3.0, scipy.where(c=='mw', 0.0, 99.0)))))) #M here defines the heat rate and emissions data we will give to generators that were not online in the historical data
            #if we are using hist_downtime, then replace scipy.NaN with M. That way offline generators can still be dispatched, but they will have high cost and high emissions.
            if self.hist_downtime:
                temp_3 = temp_3.fillna(M)
            #if we are not using hist_downtime, then use ffill to populate the scipy.NaN values. This allows us to use the last observed value for the generator to populate data that we don't have for it. For example, if generator G had a heat rate of 8.5 during time t-1, but we don't have data for time t, then we assume that generator G has a heat rate of 8.5 for t. When we do this, we can begin to include generators that might be available for dispatch but were not turned on because prices were too low. However, we also remove any chance of capturing legitimate maintenance downtime that would impact the historical data. So, for validation purposes, we probably want to have hist_downtime = True. For future scenario analysis, we probably want to have hist_downtime = False.
            if not self.hist_downtime:
                temp_3 = temp_3.fillna(method='ffill')
                temp_3.iloc[0] = temp_3.iloc[0].fillna(method='ffill') #for some reason the first row was not doing fillna(ffill)
            #merge temp_3 with df_orispl_unit. Now we have weekly heat rates, emissions rates, and capacities for each generator. These values depend on whether we are including hist_downtime
            df_orispl_unit = df_orispl_unit.merge(temp_3, on='orispl_unit', how='left')
        #merge df_orispl_unit into df. Now we have a dataframe with weekly heat rate and emissions rates for any plants in CEMS with that data. There will be some nan values in df for those weekly columns (e.g. 'heat_rate1', 'co223', etc. that we will want to fill with annual averages from eGrid for now
        orispl_units_egrid = df.orispl_unit.unique()
        orispl_units_cems = df_orispl_unit.orispl_unit.unique()
        df_leftovers = df[df.orispl_unit.isin(scipy.setdiff1d(orispl_units_egrid, orispl_units_cems))]
        #if we're doing a cems validation run, we only want to include generators that are in the CEMS data
        if self.cems_validation_run:
            df_leftovers = df_leftovers[df_leftovers.orispl_unit.isin(orispl_units_cems)]
        #remove any outliers - fuel is solar, wind, waste-heat, purchased steam, or other, less than 25MW capacity, less than 88 operating hours (1% CF), mw = nan, mmbtu = nan
        df_leftovers = df_leftovers[(df_leftovers.fuel!='SUN') & (df_leftovers.fuel!='WND') & (df_leftovers.fuel!='WH') & (df_leftovers.fuel!='OTH') & (df_leftovers.fuel!='PUR') & (df_leftovers.mw >=25) & (df_leftovers.hours_on >=88) & (~df_leftovers.mw.isna()) & (~df_leftovers.mmbtu_ann.isna())]
        #remove any outliers that have 0 emissions (except for nuclear)
        df_leftovers = df_leftovers[~((df_leftovers.fuel!='NUC') & (df_leftovers.nox_ann.isna()))]
        df_leftovers['cf'] = df_leftovers.mwh_ann / (df_leftovers.mw *8760.)
        #drop anything with capacity factor less than 1%
        df_leftovers = df_leftovers[df_leftovers.cf >= 0.01]
        df_leftovers.fillna(0.0, inplace=True)
        df_leftovers['heat_rate'] = df_leftovers.mmbtu_ann / df_leftovers.mwh_ann
        #add in the weekly time columns for heat rate and emissions rates. In this case we will just apply the annual average to each column, but we still need those columns to be able to concatenate back with df_orispl_unit and have our complete set of generator data
        for e in ['heat_rate', 'co2', 'so2', 'nox', 'mw']:
            for t in scipy.arange(52)+1:
                if e == 'mw':
                    if self.hist_downtime:
                        df_leftovers[e + str(t)] = df_leftovers[e]
                    if not self.hist_downtime:
                        df_leftovers[e + str(t)] = df_leftovers[e].quantile(0.99)
                else:
                    df_leftovers[e + str(t)] = df_leftovers[e]
        df_leftovers.drop(columns = ['gen', 'unit', 'prime_mover', 'fuel', 'mmbtu_ann', 'nox_ann', 'so2_ann', 'co2_ann', 'mwh_ann', 'fuel_type', 'co2', 'so2', 'nox', 'cf', 'heat_rate', 'hours_on', 'year_online'], inplace=True)
        #concat df_leftovers and df_orispl_unit
        df_orispl_unit = pandas.concat([df_orispl_unit, df_leftovers])
        #use df to get prime_mover, fuel, and fuel_type for each orispl_unit
        df_fuel = df[df.orispl_unit.isin(df_orispl_unit.orispl_unit.unique())][['orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'year_online']]
        df_fuel.fuel = df_fuel.fuel.str.lower()
        df_fuel.fuel_type = df_fuel.fuel_type.str.lower()
        df_fuel.prime_mover = df_fuel.prime_mover.str.lower()
        df_orispl_unit = df_orispl_unit.merge(df_fuel, on='orispl_unit', how='left')
        #if we are using, for example, 2017 CEMS and 2016 eGrid, there may be some powerplants without fuel, fuel_type, prime_mover, and year_online data. Lets assume 'ng', 'gas', 'ct', and 2017 for these units based on trends on what was built in 2017
        df_orispl_unit.loc[df_orispl_unit.fuel.isna(), ['fuel', 'fuel_type']] = ['ng', 'gas']
        df_orispl_unit.loc[df_orispl_unit.prime_mover.isna(), 'prime_mover'] = 'ct'
        df_orispl_unit.loc[df_orispl_unit.year_online.isna(), 'year_online'] = 2017
        #also change 'og' to fuel_type 'gas' instead of 'ofsl' (other fossil fuel)
        df_orispl_unit.loc[df_orispl_unit.fuel=='og', ['fuel', 'fuel_type']] = ['og', 'gas']
        df_orispl_unit.fillna(0.0, inplace=True)
        #add in some columns to aid in calculating the fuel mix
        for f_type in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
            df_orispl_unit['is_'+f_type.lower()] = (df_orispl_unit.fuel_type==f_type).astype(int)
        ###
        #derate any CHP units according to their ratio of electric fuel consumption : total fuel consumption
        #now use EIA Form 923 to flag any CHP units and calculate their ratio of total fuel : fuel used for electricity. We will use those ratios to de-rate the mw and emissions of any generators that have a CHP-flagged orispl
        #calculate the elec_ratio that is used for CHP derating
        chp_derate_df = self.eia923_1.copy(deep=True)
        chp_derate_df = chp_derate_df[(chp_derate_df.orispl.isin(df_orispl_unit.orispl)) & (chp_derate_df['Combined Heat And\nPower Plant']=='Y')].replace('.', 0.0)
        chp_derate_df = chp_derate_df[['orispl', 'Reported\nFuel Type Code', 'Elec_Quantity\nJanuary', 'Elec_Quantity\nFebruary', 'Elec_Quantity\nMarch', 'Elec_Quantity\nApril', 'Elec_Quantity\nMay', 'Elec_Quantity\nJune', 'Elec_Quantity\nJuly', 'Elec_Quantity\nAugust', 'Elec_Quantity\nSeptember', 'Elec_Quantity\nOctober', 'Elec_Quantity\nNovember', 'Elec_Quantity\nDecember', 'Quantity\nJanuary', 'Quantity\nFebruary', 'Quantity\nMarch', 'Quantity\nApril', 'Quantity\nMay', 'Quantity\nJune', 'Quantity\nJuly', 'Quantity\nAugust', 'Quantity\nSeptember', 'Quantity\nOctober', 'Quantity\nNovember', 'Quantity\nDecember']].groupby(['orispl', 'Reported\nFuel Type Code'], as_index=False).agg('sum')
        chp_derate_df['elec_ratio'] = (chp_derate_df[['Elec_Quantity\nJanuary', 'Elec_Quantity\nFebruary', 'Elec_Quantity\nMarch', 'Elec_Quantity\nApril', 'Elec_Quantity\nMay', 'Elec_Quantity\nJune', 'Elec_Quantity\nJuly', 'Elec_Quantity\nAugust', 'Elec_Quantity\nSeptember', 'Elec_Quantity\nOctober', 'Elec_Quantity\nNovember', 'Elec_Quantity\nDecember']].sum(axis=1) / chp_derate_df[['Quantity\nJanuary', 'Quantity\nFebruary', 'Quantity\nMarch', 'Quantity\nApril', 'Quantity\nMay', 'Quantity\nJune', 'Quantity\nJuly', 'Quantity\nAugust', 'Quantity\nSeptember', 'Quantity\nOctober', 'Quantity\nNovember', 'Quantity\nDecember']].sum(axis=1))
        chp_derate_df = chp_derate_df[['orispl', 'Reported\nFuel Type Code', 'elec_ratio']].dropna()
        chp_derate_df.columns = ['orispl', 'fuel', 'elec_ratio']
        chp_derate_df.fuel = chp_derate_df.fuel.str.lower()
        mw_cols = ['mw','mw1','mw2','mw3','mw4','mw5','mw6','mw7','mw8','mw9','mw10','mw11','mw12','mw13','mw14','mw15','mw16','mw17','mw18','mw19','mw20','mw21','mw22','mw23','mw24','mw25','mw26','mw27','mw28','mw29','mw30','mw31','mw32','mw33','mw34','mw35','mw36','mw37','mw38','mw39','mw40','mw41','mw42','mw43','mw44','mw45','mw46','mw47','mw48','mw49','mw50', 'mw51', 'mw52']
        chp_derate_df = df_orispl_unit.merge(chp_derate_df, how='right', on=['orispl', 'fuel'])[mw_cols + ['orispl', 'fuel', 'elec_ratio', 'orispl_unit']]
        chp_derate_df[mw_cols] = chp_derate_df[mw_cols].multiply(chp_derate_df.elec_ratio, axis='index')
        chp_derate_df.dropna(inplace=True)
        #merge updated mw columns back into df_orispl_unit
        #update the chp_derate_df index to match df_orispl_unit
        chp_derate_df.index = df_orispl_unit[df_orispl_unit.orispl_unit.isin(chp_derate_df.orispl_unit)].index
        df_orispl_unit.update(chp_derate_df[mw_cols])
        #replace the global dataframes
        self.df_cems = df_cems
        self.df = df_orispl_unit



    def calcFuelPrices(self):
        """
        let RC be a high-ish price (1.1 * SUB)
        let LIG be based on national averages
        let NG be based on purchase type, where T takes C purchase type values and nan takes C, S, & T purchase type values
        ---
        Adds one column for each week of the year to self.df that contain fuel prices for each generation unit
        """
        #we use eia923, where generators report their fuel purchases
        df = self.eia923.copy(deep=True)
        df = df[['YEAR','MONTH','orispl','ENERGY_SOURCE','FUEL_GROUP','QUANTITY','FUEL_COST', 'Purchase Type']]
        df.columns = ['year', 'month', 'orispl' , 'fuel', 'fuel_type', 'quantity', 'fuel_price', 'purchase_type']
        df.fuel = df.fuel.str.lower()
        rfc_fuel_df = pandas.read_csv('raw_data/RFC_fuel_price.csv')
        #clean up prices
        df.loc[df.fuel_price=='.', 'fuel_price'] = scipy.nan
        df.fuel_price = df.fuel_price.astype('float')/100.
        df = df.reset_index()
        #find unique monthly prices per orispl and fuel type
        #create empty dataframe to hold the results
        df2 = self.df.copy(deep=True)[['fuel','orispl','orispl_unit']]
        orispl_prices = pandas.DataFrame(columns=['orispl_unit', 'orispl', 'fuel', 1,2,3,4,5,6,7,8,9,10,11,12, 'quantity'])
        orispl_prices[['orispl_unit','orispl','fuel']] = df2[['orispl_unit', 'orispl', 'fuel']]
        #populate the results by looping through the orispl_units to see if they have EIA923 fuel price data
        for o_u in orispl_prices.orispl_unit.unique():
            #grab 'fuel' and 'orispl'
            f = orispl_prices.loc[orispl_prices.orispl_unit==o_u].iloc[0]['fuel']
            o = orispl_prices.loc[orispl_prices.orispl_unit==o_u].iloc[0]['orispl']
            #find the weighted average monthly fuel price matching 'f' and 'o'
            temp = df[(df.orispl==o) & (df.fuel==f)][['month', 'quantity', 'fuel_price']]
            if len(temp) != 0:
                temp['weighted'] = scipy.multiply(temp.quantity, temp.fuel_price)
                temp = temp.groupby(['month'], as_index=False).sum()[['month', 'quantity', 'weighted']]
                temp['fuel_price'] = scipy.divide(temp.weighted, temp.quantity)
                temp_prices = pandas.DataFrame({'month': scipy.arange(12)+1})
                temp_prices = temp_prices.merge(temp[['month', 'fuel_price']], on='month', how='left')
                temp_prices.loc[temp_prices.fuel_price.isna(), 'fuel_price'] = temp_prices.fuel_price.median()
                #add the monthly fuel prices into orispl_prices
                orispl_prices.loc[orispl_prices.orispl_unit==o_u, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel'])] = scipy.append(scipy.array(temp_prices.fuel_price),temp.quantity.sum())

        #add in additional purchasing information for slicing that we can remove later on
        orispl_prices = orispl_prices.merge(df[['orispl' , 'fuel', 'purchase_type']].drop_duplicates(subset=['orispl', 'fuel'], keep='first'), on=['orispl', 'fuel'], how='left')

        #for any fuels that we have non-zero region level EIA923 data, apply those monthly fuel price profiles to other generators with the same fuel type but that do not have EIA923 fuel price data
        f_iter = list(orispl_prices[orispl_prices[1] != 0].dropna().fuel.unique())
        if 'rc' in orispl_prices.fuel.unique():
            f_iter.append('rc')
        for f in f_iter:
            orispl_prices_filled = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices[1] != 0)].dropna().drop_duplicates(subset='orispl', keep='first').sort_values('quantity', ascending=0)
            #orispl_prices_empty = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices[1].isna())]
            orispl_prices_empty = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices[1]==0)].dropna(subset=['quantity']) #plants with some EIA923 data but no prices
            orispl_prices_nan = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices['quantity'].isna())] #plants with no EIA923 data
            multiplier = 1.00

            #if lignite, use the national fuel-quantity-weighted median
            if f == 'lig':
                print('in lig branch')
                #grab the 5th - 95th percentile prices
                temp = df[(df.fuel==f) & (df.fuel_price.notna())][['month', 'quantity', 'fuel_price', 'purchase_type']]
                temp = temp[(temp.fuel_price >= temp.fuel_price.quantile(0.05)) & (temp.fuel_price <= temp.fuel_price.quantile(0.95))]
                #weight the remaining prices according to quantity purchased
                temp['weighted'] = scipy.multiply(temp.quantity, temp.fuel_price)
                temp = temp.groupby(['month'], as_index=False).sum()[['month', 'quantity', 'weighted']]
                temp['fuel_price'] = scipy.divide(temp.weighted, temp.quantity)
                #build a dataframe that we can insert into orispl_prices
                temp_prices = pandas.DataFrame({'month': scipy.arange(12)+1})
                temp_prices = temp_prices.merge(temp[['month', 'fuel_price']], on='month', how='left')
                temp_prices.loc[temp_prices.fuel_price.isna(), 'fuel_price'] = temp_prices.fuel_price.median()
                #update orispl_prices for any units in orispl_prices_empty or orispl_prices_nan
                orispl_prices.loc[(orispl_prices.fuel==f) & ((orispl_prices.orispl.isin(orispl_prices_empty.orispl)) | (orispl_prices.orispl.isin(orispl_prices_nan.orispl))), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'purchase_type'])] = scipy.append(scipy.array(temp_prices.fuel_price),temp.quantity.sum())

        	#if natural gas, sort by supplier type (contract, tolling, spot, or other)
            elif f =='ng':
                print('in ng clause')
                orispl_prices_filled_0 = orispl_prices_filled.copy()
                orispl_prices_empty_0 = orispl_prices_empty.copy()
                #loop through the different purchase types and update any empties
                for pt in ['T', 'S', 'C']:
                    orispl_prices_filled = orispl_prices_filled_0[orispl_prices_filled_0.purchase_type==pt]
                    # [AMY] added
                    if (len(orispl_prices_filled) == 0):
                        orispl_prices_filled = orispl_prices_filled_0


                    orispl_prices_empty = orispl_prices_empty_0[orispl_prices_empty_0.purchase_type==pt]
                    multiplier = 1.00
                    #if pt == tolling prices, use a cheaper form of spot prices
                    if pt == 'T':
                        # [AMY] use any type of purchase price I can find
                        # [AMY] orispl_prices_filled = orispl_prices_filled_0[orispl_prices_filled_0.purchase_type=='S']
                        multiplier = 0.90
                    #of the plants with EIA923 data that we are assigning to plants without eia923 data, we will use the plant with the highest energy production first, assigning its fuel price profile to one of the generators that does not have EIA923 data. We will move on to plant with the next highest energy production and so on, uniformly distributing the available EIA923 fuel price profiles to generators without fuel price data
                    loop = 0
                    loop_len = len(orispl_prices_filled) - 1
                    for o in orispl_prices_empty.orispl.unique():

                        orispl_prices.loc[(orispl_prices.orispl==o) & (orispl_prices.fuel==f), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])] = scipy.array(orispl_prices_filled[orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])].iloc[loop]) * multiplier
                        #keep looping through the generators with eia923 price data until we have used all of their fuel price profiles, then start again from the beginning of the loop with the plant with the highest energy production
                        if loop < loop_len:
                            loop += 1
                        else:
                            loop = 0
                #for nan prices (those without any EIA923 information) use Spot, Contract, and Tolling Prices (i.e. all of the non-nan prices)
                #update orispl_prices_filled to include the updated empty prices
                orispl_prices_filled_new = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices[1] != 0.0)].dropna().drop_duplicates(subset='orispl', keep='first').sort_values('quantity', ascending=0)
                #loop through the filled prices and use them for nan prices
                loop = 0
                loop_len = len(orispl_prices_filled_new) - 1
                for o in orispl_prices_nan.orispl.unique():
                    orispl_prices.loc[(orispl_prices.orispl==o) & (orispl_prices.fuel==f), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])] = scipy.array(orispl_prices_filled_new[orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])].iloc[loop])
                    #keep looping through the generators with eia923 price data until we have used all of their fuel price profiles, then start again from the beginning of the loop with the plant with the highest energy production
                    if loop < loop_len:
                        loop += 1
                    else:
                        loop = 0
            #otherwise
            else:
                multiplier = 1.00
                #if refined coal, use subbit prices * 1.15
                # Use prices from RFC
                # if f =='rc':
                #     orispl_prices_filled = orispl_prices[(orispl_prices.fuel=='sub') & (orispl_prices[1] != 0.0)].dropna().drop_duplicates(subset='orispl', keep='first').sort_values('quantity', ascending=0)
                #     if (len(orispl_prices_filled) == 0):
                #     	rfc_fuel_df[(rfc_fuel_df.fuel=='sub')].dropna().drop_duplicates(subset='orispl', keep='first').sort_values('quantity', ascending=0)

                #     multiplier = 1.1
                loop = 0
                loop_len = len(orispl_prices_filled) - 1
                #of the plants with EIA923 data that we are assigning to plants without eia923 data, we will use the plant with the highest energy production first, assigning its fuel price profile to one of the generators that does not have EIA923 data. We will move on to plant with the next highest energy production and so on, uniformly distributing the available EIA923 fuel price profiles to generators without fuel price data
                if (f != 'rc'):
                    for o in scipy.concatenate((orispl_prices_empty.orispl.unique(),orispl_prices_nan.orispl.unique())):

                        orispl_prices_filled[orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])].iloc[loop]
                        scipy.array(orispl_prices_filled[orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])].iloc[loop]) * multiplier

                        orispl_prices.loc[(orispl_prices.orispl==o) & (orispl_prices.fuel==f), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])]

                        orispl_prices.loc[(orispl_prices.orispl==o) & (orispl_prices.fuel==f), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])] = scipy.array(orispl_prices_filled[orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])].iloc[loop]) * multiplier
                        #keep looping through the generators with eia923 price data until we have used all of their fuel price profiles, then start again from the beginning of the loop with the plant with the highest energy production
                        if loop < loop_len:
                            loop += 1
                        else:
                            loop = 0

        #and now we still have some nan values for fuel types that had no nerc_region eia923 data. We'll start with the national median for the EIA923 data.
        f_array = scipy.intersect1d(str(orispl_prices[orispl_prices[1].isna()].fuel.unique()), str(df.fuel.unique()))
        for f in f_array:
            temp = df[df.fuel==f][['month', 'quantity', 'fuel_price']]
            temp['weighted'] = scipy.multiply(temp.quantity, temp.fuel_price)
            temp = temp.groupby(['month'], as_index=False).sum()[['month', 'quantity', 'weighted']]
            temp['fuel_price'] = scipy.divide(temp.weighted, temp.quantity)
            temp_prices = pandas.DataFrame({'month': scipy.arange(12)+1})
            temp_prices = temp_prices.merge(temp[['month', 'fuel_price']], on='month', how='left')
            temp_prices.loc[temp_prices.fuel_price.isna(), 'fuel_price'] = temp_prices.fuel_price.median()
            orispl_prices.loc[orispl_prices.fuel==f, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'purchase_type'])] = scipy.append(scipy.array(temp_prices.fuel_price),temp.quantity.sum())
        #for any fuels that don't have EIA923 data at all (for all regions) we will use commodity price approximations from an excel file
        #first we need to change orispl_prices from months to weeks
        orispl_prices.columns = ['orispl_unit', 'orispl', 'fuel', 1, 5, 9, 14, 18, 22, 27, 31, 36, 40, 44, 48, 'quantity', 'purchase_type']
        #scipy.array(orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type']))
        test = orispl_prices.copy(deep=True)[['orispl_unit', 'orispl', 'fuel']]
        month_weeks = scipy.array(orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type']))
        for c in scipy.arange(52)+1:
            if c in month_weeks:
                test['fuel_price'+ str(c)] = orispl_prices[c]
            else:
                test['fuel_price'+ str(c)] = test['fuel_price'+str(c-1)]
        orispl_prices = test.copy(deep=True)
        #now we add in the weekly fuel commodity prices
        prices_fuel_commodity = self.fuel_commodity_prices
        f_array = orispl_prices[orispl_prices['fuel_price1'].isna()].fuel.unique()
        for f in f_array:
            l = len(orispl_prices.loc[orispl_prices.fuel==f, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel'])])
            orispl_prices.loc[orispl_prices.fuel==f, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel'])] = scipy.tile(prices_fuel_commodity[f], (l,1))

        # [AMY] Handle subbit prices by grabbing the most expensive data from RFC_fuel_price
        rc_orispl_unit = "1743_3"
        if (len(orispl_prices[orispl_prices.fuel=='rc'])!= 0):
            orispl_prices.loc[orispl_prices.fuel=='rc', orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel'])] = rfc_fuel_df.loc[rfc_fuel_df.orispl_unit==rc_orispl_unit, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel'])]

        #now we have orispl_prices, which has a weekly fuel price for each orispl_unit based mostly on EIA923 data with some commodity, national-level data from EIA to supplement
        #now merge the fuel price columns into self.df
        orispl_prices.drop(['orispl', 'fuel'], axis=1, inplace=True)


        #save
        self.df = self.df.merge(orispl_prices, on='orispl_unit', how='left')



    def easiurDamages(self):
        """
        Adds EASIUR environmental damages for SO2 and NOx emissions for each power plant.
        ---
        Adds one column for each week of the year to self.df that contains environmental damages in $/MWh for each generation unit calculated using the EASIURE method
        """
        print ('Adding environmental damages...')
        #clean the easiur data
        df = self.easiur_per_plant.copy(deep=True)
        df = df[['ORISPL','SO2 Winter 150m','SO2 Spring 150m','SO2 Summer 150m','SO2 Fall 150m','NOX Winter 150m','NOX Spring 150m','NOX Summer 150m','NOX Fall 150m']]
        df.columns = ['orispl', 'so2_dmg_win', 'so2_dmg_spr' , 'so2_dmg_sum', 'so2_dmg_fal', 'nox_dmg_win', 'nox_dmg_spr' , 'nox_dmg_sum', 'nox_dmg_fal']
        #create empty dataframe to hold the results
        df2 = self.df.copy(deep=True)
        #for each week, calculate the $/MWh damages for each generator based on its emissions rate (kg/MWh) and easiur damages ($/tonne)
        for c in scipy.arange(52)+1:
            season = scipy.where(((c>49) | (c<=10)), 'win', scipy.where(((c>10) & (c<=23)), 'spr', scipy.where(((c>23) & (c<=36)), 'sum', scipy.where(((c>36) & (c<=49)), 'fal', 'na')))) #define the season string
            df2['dmg' + str(c)] = (df2['so2' + str(c)] * df['so2' + '_dmg_' + str(season)] + df2['nox' + str(c)] * df['nox' + '_dmg_' + str(season)]) / 1e3
        #use the results to redefine the main generator DataFrame
        self.df = df2


    def addGenMinOut(self):
        """
        Adds fuel price and vom costs to the generator dataframe 'self.df'
        ---
        """
        df = self.df.copy(deep=True)
        #define min_out, based on the NREL Black & Veatch report (2012)
        min_out_coal = 0.4
        min_out_ngcc = 0.5
        min_out_ngst = min_out_coal #assume the same as coal boiler
        min_out_nggt = 0.5
        min_out_oilst = min_out_coal #assume the same as coal boiler
        min_out_oilgt = min_out_nggt #assume the same as gas turbine
        min_out_nuc = 0.5
        min_out_bio = 0.4
        df['min_out_multiplier'] = scipy.where(df.fuel_type=='oil', scipy.where(df.prime_mover=='st', min_out_oilst, min_out_oilgt), scipy.where(df.fuel_type=='biomass',min_out_bio, scipy.where(df.fuel_type=='coal',min_out_coal, scipy.where(df.fuel_type=='nuclear',min_out_nuc, scipy.where(df.fuel_type=='gas', scipy.where(df.prime_mover=='gt', min_out_nggt, scipy.where(df.prime_mover=='st', min_out_ngst, min_out_ngcc)), 0.10)))))
        df['min_out'] = df.mw * df.min_out_multiplier
        self.df = df


    def addGenVom(self):
        """
        Adds vom costs to the generator dataframe 'self.df'
        ---
        """
        df = self.df.copy(deep=True)
        #define vom, based on the ranges of VOM values from pg.12, fig 5 of NREL The Western Wind and Solar Integration Study Phase 2" report. We assume, based on that study, that older units have higher values and newer units have lower values according to a linear relationship between the following coordinates:
        vom_range_coal_bit = [1.5, 5]
        vom_range_coal_sub = [1.5, 5]
        vom_range_coal = [1.5, 5]
        age_range_coal = [1955, 2013]
        vom_range_ngcc = [0.5, 1.5]
        age_range_ngcc = [1990, 2013]
        vom_range_nggt = [0.5, 2.0]
        age_range_nggt = [1970, 2013]
        vom_range_ngst = [0.5, 6.0]
        age_range_ngst = [1950, 2013]

        def vom_calculator(fuelType, fuel, primeMover, yearOnline):
            if fuelType=='coal':
                if fuel == 'bit':
                    return vom_range_coal_bit[0] + (vom_range_coal_bit[1]-vom_range_coal_bit[0])/(age_range_coal[1]-age_range_coal[0]) * (self.year - yearOnline)
                elif fuel == 'sub':
                    return vom_range_coal_sub[0] + (vom_range_coal_sub[1]-vom_range_coal_sub[0])/(age_range_coal[1]-age_range_coal[0]) * (self.year - yearOnline)
                else:
                    return vom_range_coal[0] + (vom_range_coal[1]-vom_range_coal[0])/(age_range_coal[1]-age_range_coal[0]) * (self.year - yearOnline)
            if fuelType!='coal':
                if (primeMover=='ct') | (primeMover=='cc'):
                    return vom_range_ngcc[0] + (vom_range_ngcc[1]-vom_range_ngcc[0])/(age_range_ngcc[1]-age_range_ngcc[0]) * (self.year - yearOnline)
                if primeMover=='gt':
                    return vom_range_nggt[0] + (vom_range_nggt[1]-vom_range_nggt[0])/(age_range_nggt[1]-age_range_nggt[0]) * (self.year - yearOnline)
                if primeMover=='st':
                    return vom_range_ngst[0] + (vom_range_ngst[1]-vom_range_ngst[0])/(age_range_ngst[1]-age_range_ngst[0]) * (self.year - yearOnline)

        df['vom'] = df.apply(lambda x: vom_calculator(x['fuel_type'], x['fuel'], x['prime_mover'], x['year_online']), axis=1)
        self.df = df


    def addDummies(self):
        """
        Adds dummy "coal_0" and "ngcc_0" generators to df
        ---
        """
        df = self.df.copy(deep=True)
        #coal_0
        df.loc[len(df)] = df.loc[0]
        df.loc[len(df)-1, self.df.columns.drop(['ba', 'isorto', 'egrid'])] = df.loc[0, df.columns.drop(['ba', 'isorto', 'egrid'])] * 0
        df.loc[len(df)-1,['orispl', 'orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'min_out_multiplier', 'min_out', 'is_coal']] = ['coal_0', 'coal_0', 'sub', 'coal', 'st', 0.0, 0.0, 1]
        #ngcc_0
        df.loc[len(df)] = df.loc[0]
        df.loc[len(df)-1, self.df.columns.drop(['ba', 'isorto', 'egrid'])] = df.loc[0, df.columns.drop(['ba', 'isorto', 'egrid'])] * 0
        df.loc[len(df)-1,['orispl', 'orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'min_out_multiplier', 'min_out', 'is_gas']] = ['ngcc_0', 'ngcc_0', 'ng', 'gas', 'ct', 0.0, 0.0, 1]
        self.df = df


    def calcDemandData(self, demand):
        """
        Uses CEMS data to calculate net demand (i.e. total fossil generation), total emissions, and each generator type's contribution to the generation mix
        ---
        Creates
        self.hist_dispatch : one row per hour of the year, columns for net demand, total emissions, operating cost of the marginal generator, and the contribution of different fuels to the total energy production
        """
        print ('Calculating demand data from CEMS...')
        #re-compile the cems data adding in fuel and fuel type
        df = self.df_cems.copy(deep=True)
        merge_orispl_unit = self.df.copy(deep=True)[['orispl_unit', 'fuel', 'fuel_type']]
        merge_orispl = self.df.copy(deep=True)[['orispl', 'fuel', 'fuel_type']].drop_duplicates('orispl')
        df = df.merge(merge_orispl_unit, left_index=True, how='left', on=['orispl_unit'])
        df.loc[df.fuel.isna(), 'fuel'] = scipy.array(df[df.fuel.isna()].merge(merge_orispl, left_index=True, how='left', on=['orispl']).fuel_y)
        df.loc[df.fuel_type.isna(), 'fuel_type'] = scipy.array(df[df.fuel_type.isna()].merge(merge_orispl, left_index=True, how='left', on=['orispl']).fuel_type_y)
        #build the hist_dispatch dataframe
        #start with the datetime column
        start_date_str = (self.df_cems.date.min()[-4:] + '-' + self.df_cems.date.min()[:5] + ' 00:00')
        date_hour_count = len(self.df_cems.date.unique())*24#+1
        hist_dispatch = pandas.DataFrame(scipy.array([pandas.Timestamp(start_date_str) + datetime.timedelta(hours=i) for i in xrange(date_hour_count)]), columns=['datetime'])
        
        #add columns by aggregating df by date + hour
        #hist_dispatch['demand'] = df.groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['co2_tot'] = df.groupby(['date','hour'], as_index=False).sum().co2_tot # * 2000
        hist_dispatch['so2_tot'] = df.groupby(['date','hour'], as_index=False).sum().so2_tot
        hist_dispatch['nox_tot'] = df.groupby(['date','hour'], as_index=False).sum().nox_tot
        hist_dispatch['coal_mix'] = df[(df.fuel_type=='coal') | (df.fuel=='SGC')].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['gas_mix'] = df[df.fuel_type=='gas'].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['oil_mix'] = df[df.fuel_type=='oil'].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['biomass_mix'] = df[(df.fuel_type=='biomass') | (df.fuel=='obs') | (df.fuel=='wds') | (df.fuel=='blq') | (df.fuel=='msw') | (df.fuel=='lfg') | (df.fuel=='ab') | (df.fuel=='obg') | (df.fuel=='obl') | (df.fuel=='slw')].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['geothermal_mix'] = df[(df.fuel_type=='geothermal') | (df.fuel=='geo')].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['hydro_mix'] = df[(df.fuel_type=='hydro') | (df.fuel=='wat')].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['nuclear_mix'] = df[df.fuel=='nuc'].groupby(['date','hour'], as_index=False).sum().mwh
        
        # Add demand here 
        demand.columns = ['datetime', 'demand']
        demand.datetime = pandas.to_datetime(demand.datetime)
        hist_dispatch = hist_dispatch.merge(demand, how='inner', on='datetime')

        hist_dispatch.fillna(0, inplace=True)
        #fill in last line to equal the previous line
        #hist_dispatch.loc[(len(hist_dispatch)-1)] = hist_dispatch.loc[(len(hist_dispatch)-2)]
        self.hist_dispatch = hist_dispatch


    def addElecPriceToDemandData(self):
        """
        Calculates the historical electricity price for the nerc region, adding it as a new column to the demand data
        ---
        """
        print ('Adding historical electricity prices...')
        #We will use FERC 714 data, where balancing authorities and similar entities report their locational marginal prices. This script pulls in those price for every reporting entity in the nerc region and takes the max price across the BAs/entities for each hour.
        df = self.ferc714.copy(deep=True)
        df_ids = self.ferc714_ids.copy(deep=True)
        nerc_region = self.isorto
        year = self.year
        df_ids_bas = list(df_ids[df_ids.isorto == nerc_region].respondent_id.values)
        #aggregate the price data by mean price per hour for any balancing authorities within the nerc region
        df_bas = df[df.respondent_id.isin(df_ids_bas) & (df.report_yr==year)][['lambda_date', 'respondent_id', 'hour01', 'hour02', 'hour03', 'hour04', 'hour05', 'hour06', 'hour07', 'hour08', 'hour09', 'hour10', 'hour11', 'hour12', 'hour13', 'hour14', 'hour15', 'hour16', 'hour17', 'hour18', 'hour19', 'hour20', 'hour21', 'hour22', 'hour23', 'hour24']]
        df_bas.drop(['respondent_id'], axis=1, inplace=True)
        df_bas.columns = ['date',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        df_bas_temp = pandas.melt(df_bas, id_vars=['date'])
        df_bas_temp.date = df_bas_temp.date.str[0:-7] + (df_bas_temp.variable - 1).astype(str) + ':00'
        df_bas_temp['time'] = df_bas_temp.variable.astype(str) + ':00'
        df_bas_temp['datetime'] = pandas.to_datetime(df_bas_temp.date)
        df_bas_temp.drop(columns=['date', 'variable', 'time'], inplace=True)
        #aggregate by datetime
        df_bas_temp = df_bas_temp.groupby('datetime', as_index=False).max()
        df_bas_temp.columns = ['datetime', 'price']
        #add the price column to self.hist_dispatch
        self.hist_dispatch['gen_cost_marg'] = df_bas_temp.price


    def demandTimeSeries(self, demand):
        """
        Re-formats and slices self.hist_dispatch to produce a demand time series to be used by the dispatch object
        ---
        Creates
        self.demand_data : row for each hour. columns for datetime and demand
        """
        print ('Creating "demand_data" time series...')
        #demand using CEMS data
        demand.columns = ['datetime', 'demand']
        demand.datetime = pandas.to_datetime(demand.datetime)
        self.demand_data = demand


    def calcMdtCoalEvents(self):
        """
        Creates a dataframe of the start, end, and demand_threshold for each event in the demand data where we would expect a coal plant's minimum downtime constraint to kick in
        ---
        """
        mdt_coal_events = self.demand_data.copy()
        mdt_coal_events['indices'] = mdt_coal_events.index
        #find the integral from x to x+
        mdt_coal_events['integral_x_xt'] = mdt_coal_events.demand[::-1].rolling(window=self.coal_min_downtime+1).sum()[::-1]
        #find the integral of a flat horizontal line extending from x
        mdt_coal_events['integral_x'] = mdt_coal_events.demand * (self.coal_min_downtime+1)
        #find the integral under the minimum of the flat horizontal line and the demand curve
        def d_forward_convex_integral(mdt_index):
            try:
                return scipy.minimum(scipy.repeat(mdt_coal_events.demand[mdt_index], (self.coal_min_downtime+1)), mdt_coal_events.demand[mdt_index:(mdt_index+self.coal_min_downtime+1)]).sum()
            except:
                return mdt_coal_events.demand[mdt_index]
        mdt_coal_events['integral_x_xt_below_x'] = mdt_coal_events.indices.apply(d_forward_convex_integral)
        #find the integral of the convex portion below x_xt
        mdt_coal_events['integral_convex_portion_btwn_x_xt'] = mdt_coal_events['integral_x'] - mdt_coal_events['integral_x_xt_below_x']
        #keep the convex integral only if x < 1.05*x+
        def d_keep_convex(mdt_index):
            try:
                return int(mdt_coal_events.demand[mdt_index] <= 1.05*mdt_coal_events.demand[mdt_index + self.coal_min_downtime]) * mdt_coal_events.integral_convex_portion_btwn_x_xt[mdt_index]
            except:
                return mdt_coal_events.integral_convex_portion_btwn_x_xt[mdt_index]
        mdt_coal_events['integral_convex_filtered'] = mdt_coal_events.indices.apply(d_keep_convex)
        #mdt_coal_events['integral_convex_filtered'] = mdt_coal_events['integral_convex_filtered'].replace(0, scipy.nan)
        #keep any local maximums of the filtered convex integral
        mdt_coal_events['local_maximum'] = ((mdt_coal_events.integral_convex_filtered== mdt_coal_events.integral_convex_filtered.rolling(window=self.coal_min_downtime/2+1, center=True).max()) & (mdt_coal_events.integral_convex_filtered != 0) & (mdt_coal_events.integral_x >= mdt_coal_events.integral_x_xt))
        #spread the maximum out over the min downtime window
        mdt_coal_events = mdt_coal_events[mdt_coal_events.local_maximum]
        mdt_coal_events['demand_threshold'] = mdt_coal_events.demand
        mdt_coal_events['start'] = mdt_coal_events.datetime
        mdt_coal_events['end'] = mdt_coal_events.start + pandas.DateOffset(hours=self.coal_min_downtime)
        mdt_coal_events = mdt_coal_events[['start', 'end', 'demand_threshold']]
        self.mdt_coal_events = mdt_coal_events


if __name__ == '__main__':
    run_year = 2017
    run_type = 'forecast' # actual or forecast 
    nerc_region = 'PJM'

    ferc714IDs_csv = 'raw_data/Respondent IDs.csv'
    easiur_csv_path = 'raw_data/egrid_2016_plant_easiur.csv'
    fuel_commodity_prices_xlsx = 'raw_data/fuel_default_prices.xlsx'
    ferc714_part2_schedule6_csv = 'raw_data/Part 2 Schedule 6 - Balancing Authority Hourly System Lambda.csv'

    if run_year == 2017:
        egrid_data_xlsx = 'raw_data/egrid2016_data.xlsx' 
        eia923_schedule5_xlsx = 'raw_data/EIA923_Schedules_2_3_4_5_M_12_2017_Final_Revision.xlsx'
        cems_folder_path = 'raw_data/CEMS/2017/'
        if run_type == 'actual':
            demand_data = pandas.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', 'generation', 'actual_fossil_gen_2017.csv'))
            pjm_dispatch_save_folder = 'baseline/2017/'
        else: 
            demand_data = pandas.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'neural_net_model', 'forecasted_fossil_gen.csv'))
            pjm_dispatch_save_folder = 'forecasted/2017/'

    if run_year == 2016:
        egrid_data_xlsx = 'raw_data/egrid2016_data.xlsx'
        eia923_schedule5_xlsx = 'raw_data/EIA923_Schedules_2_3_4_5_M_12_2016_Final_Revision.xlsx'
        cems_folder_path = 'raw_data/CEMS/2016/'
        demand_data = pandas.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', 'generation', 'actual_fossil_gen_2016.csv'))
        pjm_dispatch_save_folder = 'baseline/2016/'

    #run the generator data object
    gd = generatorData(nerc_region, egrid_fname=egrid_data_xlsx, eia923_fname=eia923_schedule5_xlsx, demand = demand_data, ferc714IDs_fname=ferc714IDs_csv, ferc714_fname=ferc714_part2_schedule6_csv, cems_folder=cems_folder_path, easiur_fname=easiur_csv_path, include_easiur_damages=True, year=run_year, fuel_commodity_prices_excel_dir=fuel_commodity_prices_xlsx, hist_downtime=False, coal_min_downtime = 12, cems_validation_run=False)
    print('generator data complete!')

    #create a shortened version that has only the essentials (so we can pickle)
    gd_short = {'year': gd.year, 'nerc': gd.nerc, 'hist_dispatch': gd.hist_dispatch, 'demand_data': gd.demand_data, 'mdt_coal_events': gd.mdt_coal_events, 'df': gd.df}
    
    #save the historical dispatch
    for keys,values in gd_short.items():
        if (keys=='hist_dispatch'):
            gd_short[keys].to_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + '_hourly_demand_and_fuelmix.csv')
        if (keys=='demand_data'):
            gd_short[keys].to_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + 'demand_data.csv')
        if (keys=='mdt_coal_events'):
            gd_short[keys].to_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + 'mdt_coal_events.csv')
        if (keys=='df'):
            gd_short[keys].to_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + 'df.csv')
