# simple_dispatch
# Thomas Deetjen
# v_21
# last edited: 2019-07-02
# class "bidStack" creates a merit order curve from the generator fleet information created by the "generatorData" class
# class "dispatch" uses the "bidStack" object to choose which power plants should be operating during each time period to meet a demand time-series input

import pandas
import matplotlib.pylab
import scipy
import scipy.interpolate
import datetime
import math
import copy
from bisect import bisect_left



class bidStack(object):
    def __init__(self, gen_data_short, co2_dol_per_kg=0.0, so2_dol_per_kg=0.0, nox_dol_per_kg=0.0, coal_dol_per_mmbtu=0.0, coal_capacity_derate = 0.0, time=1, dropNucHydroGeo=False, include_min_output=True, initialization=True, coal_mdt_demand_threshold = 0.0, mdt_weight=0.50):
        """
        1) Bring in the generator data created by the "generatorData" class.
        2) Calculate the generation cost for each generator and sort the generators by generation cost. Default emissions prices [$/kg] are 0.00 for all emissions.
        ---
        gen_data_short : a generatorData object
        co2 / so2 / nox_dol_per_kg : a tax on each amount of emissions produced by each generator. Impacts each generator's generation cost
        coal_dol_per_mmbtu : a tax (+) or subsidy (-) on coal fuel prices in $/mmbtu. Impacts each generator's generation cost
        coal_capacity_derate : fraction that we want to derate all coal capacity (e.g. 0.20 mulutiplies each coal plant's capacity by (1-0.20))
        time : number denoting which time period we are interested in. Default is weeks, so time=15 would look at the 15th week of heat rate, emissions rates, and fuel prices
        dropNucHydroGeo : if True, nuclear, hydro, and geothermal plants will be removed from the bidstack (e.g. to match CEMS data)
        include_min_output : if True, will include a representation of generators' minimum output constraints that impacts the marginal generators in the dispatch. So, a "True" value here is closer to the real world.
        initialization : if True, the bs object is being defined for the first time. This will trigger the generation of a dummy 0.0 demand generator to bookend the bottom of the merit order (in calcGenCost function) after which initialization will be set to False
        """
        for keys,values in gd_short.items():
        	if (keys=='year'):
        		self.year = gen_data_short[keys]
        	elif (keys=='nerc'):
        		self.nerc = gen_data_short[keys]
        	elif (keys=='hist_dispatch'):
        		self.hist_dispatch = gen_data_short[keys]
        	elif (keys=='mdt_coal_events'):
        		self.mdt_coal_events = gen_data_short[keys]
        	elif (keys=='df'):
        		self.df_0 = gen_data_short[keys]

        self.coal_mdt_demand_threshold = coal_mdt_demand_threshold
        self.mdt_weight = mdt_weight
        self.df = self.df_0.copy(deep=True)
        self.co2_dol_per_kg = co2_dol_per_kg
        self.so2_dol_per_kg = so2_dol_per_kg
        self.nox_dol_per_kg = nox_dol_per_kg
        self.coal_dol_per_mmbtu = coal_dol_per_mmbtu
        self.coal_capacity_derate = coal_capacity_derate
        self.time = time
        self.include_min_output = include_min_output
        self.initialization = initialization
        if dropNucHydroGeo:
            self.dropNuclearHydroGeo()
        #self.addFuelColor()
        self.processData()
        #self.df.to_csv('bidstack_WOW.csv')


    def updateDf(self, new_data_frame):
        self.df_0 = new_data_frame
        self.df = self.df_0.copy(deep=True)
        self.processData()


    def dropNuclearHydroGeo(self):
        """
        Removes nuclear, hydro, and geothermal plants from self.df_0 (since they don't show up in CEMS)
        ---
        """
        self.df_0 = self.df_0[(self.df_0.fuel!='nuc') & (self.df_0.fuel!='wat') & (self.df_0.fuel!='geo')]


    def updateEmissionsAndFuelTaxes(self, co2_price_new, so2_price_new, nox_price_new, coal_price_new):
        """ Updates self. emissions prices (in $/kg) and self.coal_dol_per_mmbtu (in $/mmbtu)
        ---
        """
        self.co2_dol_per_kg = co2_price_new
        self.so2_dol_per_kg = so2_price_new
        self.nox_dol_per_kg = nox_price_new
        self.coal_dol_per_mmbtu = coal_price_new


    def processData(self):
        """ runs a few of the internal functions. There are couple of places in the class that run these functions in this order, so it made sense to just locate this set of function runs in a single location
        ---
        """
        self.calcGenCost()
        self.createTotalInterpolationFunctions()
        self.createMarginalPiecewise()
        self.calcFullMeritOrder()
        self.createMarginalPiecewise() #create this again after FullMeritOrder so that it includes the new full_####_marg columns
        self.createTotalInterpolationFunctionsFull()


    def updateTime(self, t_new):
        """ Updates self.time
        ---
        """
        self.time = t_new
        self.processData()


    def addFuelColor(self):
        """ Assign a fuel type for each fuel and a color for each fuel type to be used in charts
        ---
        creates 'fuel_type' and 'fuel_color' columns
        """
        c = {'gas':'#888888', 'coal':'#bf5b17', 'oil':'#252525' , 'nuclear':'#984ea3', 'hydro':'#386cb0', 'biomass':'#7fc97f', 'geothermal':'#e31a1c', 'ofsl': '#c994c7'}
        self.df_0['fuel_color'] = '#bcbddc'
        for c_key in c.iterkeys():
            self.df_0.loc[self.df_0.fuel_type == c_key, 'fuel_color'] = c[c_key]


    def calcGenCost(self):
        """ Calculate average costs that are function of generator data, fuel cost, and emissions prices.
        gen_cost ($/MWh) = (heat_rate * "fuel"_price) + (co2 * co2_price) + (so2 * so2_price) + (nox * nox_price) + vom
        """
        df = self.df_0.copy(deep=True)
        df.to_csv("calc_gen_cost.csv")
        #pre-processing:
            #adjust coal fuel prices by the "coal_dol_per_mmbtu" input
        df.loc[df.fuel_type=='coal', 'fuel_price' + str(self.time)] = scipy.maximum(0, df.loc[df.fuel_type=='coal', 'fuel_price' + str(self.time)] + self.coal_dol_per_mmbtu)
            #adjust coal capacity by the "coal_capacity_derate" input
        df.loc[df.fuel_type=='coal', 'mw' + str(self.time)] = df.loc[df.fuel_type=='coal', 'mw' + str(self.time)] * (1.0 -  self.coal_capacity_derate)
        #calculate the generation cost:
        df['fuel_cost'] = df['heat_rate' + str(self.time)] * df['fuel_price' + str(self.time)]
        df['co2_cost'] = df['co2' + str(self.time)] * self.co2_dol_per_kg
        df['so2_cost'] = df['so2' + str(self.time)] * self.so2_dol_per_kg
        df['nox_cost'] = df['nox' + str(self.time)] * self.nox_dol_per_kg
        df['gen_cost'] = scipy.maximum(0.01, df.fuel_cost + df.co2_cost + df.so2_cost + df.nox_cost + df.vom)
        #add a zero generator so that the bid stack goes all the way down to zero. This is important for calculating information for the marginal generator when the marginal generator is the first one in the bid stack.
        df['dmg_easiur'] = df['dmg' + str(self.time)]
        #if self.initialization:
        df = df.append(df.loc[0]*0)
        df = df.append(df.iloc[-1])
        #self.initialization = False
        df.sort_values('gen_cost', inplace=True)
        #move coal_0 and ngcc_0 to the front of the merit order regardless of their gen cost
        coal_0_ind = df[df.orispl_unit=='coal_0'].index[0]
        ngcc_0_ind = df[df.orispl_unit=='ngcc_0'].index[0]
        df = pandas.concat([df.iloc[[0],:], df[df.orispl_unit=='coal_0'], df[df.orispl_unit=='ngcc_0'], df.drop([0, coal_0_ind, ngcc_0_ind], axis=0)], axis=0)
        df.reset_index(drop=True, inplace=True)
        df['demand'] = df['mw' + str(self.time)].cumsum()
        df.loc[len(df)-1, 'demand'] = df.loc[len(df)-1, 'demand'] + 1000000 #creates a very large generator at the end of the merit order so that demand cannot be greater than supply
        df['f'] = df['demand']
        df['s'] = scipy.append(0, scipy.array(df.f[0:-1]))
        df['a'] = scipy.maximum(df.s - df.min_out*(1/0.10), 1.0)
        #add a very large demand for the last row
        self.df = df


    def createMarginalPiecewise(self):
        """ Creates a piecewsise dataframe of the generator data. We can then interpolate this data frame for marginal data instead of querying.
        """
        test = self.df.copy()
        test_shift = test.copy()
        test_shift[['demand']] = test_shift.demand + 0.1
        test.index = test.index * 2
        test_shift.index = test_shift.index * 2 + 1
        df_marg_piecewise = pandas.concat([test, test_shift]).sort_index()
        df_marg_piecewise[['demand']] = pandas.concat([df_marg_piecewise.demand[0:1], df_marg_piecewise.demand[0:-1]]).reset_index(drop=True)
        df_marg_piecewise[['demand']] = df_marg_piecewise.demand - 0.1
        self.df_marg_piecewise = df_marg_piecewise


    def returnMarginalGenerator(self, demand, return_type):
        """ Returns marginal data by interpolating self.df_marg_piecewise, which is much faster than the returnMarginalGenerator function below.
        ---
        demand : [MW]
        return_type : column header of self.df being returned (e.g. 'gen', 'fuel_type', 'gen_cost', etc.)
        """
        try: #try interpolation as it's much faster.
            try: #for columns with a time value at the end (i.e. nox30)
                return scipy.interp(demand, self.df_marg_piecewise['demand'], scipy.array(self.df_marg_piecewise[return_type + str(self.time)], dtype='float64'))
            except: #for columns without a time value at the end (i.e. gen_cost)
                return scipy.interp(demand, self.df_marg_piecewise['demand'], scipy.array(self.df_marg_piecewise[return_type], dtype='float64'))
        except: #interpolation will only work for floats, so we use querying below otherwise (~8x slower)
            ind = scipy.minimum(self.df.index[self.df.demand <= demand][-1], len(self.df)-2)
            return self.df[return_type][ind+1]


    def createTotalInterpolationFunctions(self):
        """ Creates interpolation functions for the total data (i.e. total cost, total emissions, etc.) depending on total demand. Then the returnTotalCost, returnTotal###, ..., functions use these interpolations rather than querying the dataframes as in previous versions. This reduces solve time by ~90x.
        """
        test = self.df.copy()
        #cost
        self.f_totalCost = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['gen_cost']).cumsum())
        #emissions and health damages
        self.f_totalCO2 = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['co2' + str(self.time)]).cumsum())
        self.f_totalSO2 = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['so2' + str(self.time)]).cumsum())
        self.f_totalNOX = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['nox' + str(self.time)]).cumsum())
        self.f_totalDmg = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['dmg' + str(self.time)]).cumsum())
        #for coal units only
        self.f_totalCO2_Coal = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['co2' + str(self.time)] * test['is_coal']).cumsum())
        self.f_totalSO2_Coal = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['so2' + str(self.time)] * test['is_coal']).cumsum())
        self.f_totalNOX_Coal = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['nox' + str(self.time)] * test['is_coal']).cumsum())
        self.f_totalDmg_Coal = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['dmg' + str(self.time)] * test['is_coal']).cumsum())
        #fuel mix
        self.f_totalGas = scipy.interpolate.interp1d(test.demand, (test['is_gas'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalCoal = scipy.interpolate.interp1d(test.demand, (test['is_coal'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalOil = scipy.interpolate.interp1d(test.demand, (test['is_oil'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalNuclear = scipy.interpolate.interp1d(test.demand, (test['is_nuclear'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalHydro = scipy.interpolate.interp1d(test.demand, (test['is_hydro'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalGeothermal = scipy.interpolate.interp1d(test.demand, (test['is_geothermal'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalBiomass = scipy.interpolate.interp1d(test.demand, (test['is_biomass'] * test['mw' + str(self.time)]).cumsum())
        #fuel consumption
        self.f_totalConsGas = scipy.interpolate.interp1d(test.demand, (test['is_gas'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsCoal = scipy.interpolate.interp1d(test.demand, (test['is_coal'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsOil = scipy.interpolate.interp1d(test.demand, (test['is_oil'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsNuclear = scipy.interpolate.interp1d(test.demand, (test['is_nuclear'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsHydro = scipy.interpolate.interp1d(test.demand, (test['is_hydro'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsGeothermal = scipy.interpolate.interp1d(test.demand, (test['is_geothermal'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsBiomass = scipy.interpolate.interp1d(test.demand, (test['is_biomass'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())


    def returnTotalCost(self, demand):
        """ Given demand input, return the integral of the bid stack generation cost (i.e. the total operating cost of the online power plants).
        ---
        demand : [MW]
        return : integral value of the bid stack cost = total operating costs of the online generator fleet [$].
        """
        return self.f_totalCost(demand)


    def returnTotalEmissions(self, demand, emissions_type):
        """ Given demand and emissions_type inputs, return the integral of the bid stack emissions (i.e. the total emissions of the online power plants).
        ---
        demand : [MW]
        emissions_type : 'co2', 'so2', 'nox', etc.
        return : integral value of the bid stack emissions = total emissions of the online generator fleet [lbs].
        """
        if emissions_type == 'co2':
            return self.f_totalCO2(demand)
        if emissions_type == 'so2':
            return self.f_totalSO2(demand)
        if emissions_type == 'nox':
            return self.f_totalNOX(demand)


    def returnTotalEmissions_Coal(self, demand, emissions_type):
        """ Given demand and emissions_type inputs, return the integral of the bid stack emissions (i.e. the total emissions of the online power plants).
        ---
        demand : [MW]
        emissions_type : 'co2', 'so2', 'nox', etc.
        return : integral value of the bid stack emissions = total emissions of the online generator fleet [lbs].
        """
        if emissions_type == 'co2':
            return self.f_totalCO2_Coal(demand)
        if emissions_type == 'so2':
            return self.f_totalSO2_Coal(demand)
        if emissions_type == 'nox':
            return self.f_totalNOX_Coal(demand)


    def returnTotalEasiurDamages(self, demand):
        """ Given demand input, return the integral of the bid stack EASIUR damages (i.e. the total environmental damages of the online power plants).
        ---
        demand : [MW]
        return : integral value of the bid environmental damages = total damages of the online generator fleet [$].
        """
        return self.f_totalDmg(demand)


    def returnTotalEasiurDamages_Coal(self, demand):
        """ Given demand input, return the integral of the bid stack EASIUR damages (i.e. the total environmental damages of the online power plants).
        ---
        demand : [MW]
        return : integral value of the bid environmental damages = total damages of the online generator fleet [$].
        """
        return self.f_totalDmg_Coal(demand)


    def returnTotalFuelMix(self, demand, is_fuel_type):
        """ Given demand and is_fuel_type inputs, return the total MW of online generation of is_fuel_type.
        ---
        demand : [MW]
        is_fuel_type : 'is_coal', etc.
        return : total amount of online generation of type is_fuel_type
        """
        if is_fuel_type == 'is_gas':
            return self.f_totalGas(demand)
        if is_fuel_type == 'is_coal':
            return self.f_totalCoal(demand)
        if is_fuel_type == 'is_oil':
            return self.f_totalOil(demand)
        if is_fuel_type == 'is_nuclear':
            return self.f_totalNuclear(demand)
        if is_fuel_type == 'is_hydro':
            return self.f_totalHydro(demand)
        if is_fuel_type == 'is_geothermal':
            return self.f_totalGeothermal(demand)
        if is_fuel_type == 'is_biomass':
            return self.f_totalBiomass(demand)


    def returnTotalFuelConsumption(self, demand, is_fuel_type):
        """ Given demand and is_fuel_type inputs, return the total MW of online generation of is_fuel_type.
        ---
        demand : [MW]
        is_fuel_type : 'is_coal', etc.
        return : total amount of fuel consumption of type is_fuel_type
        """
        if is_fuel_type == 'is_gas':
            return self.f_totalConsGas(demand)
        if is_fuel_type == 'is_coal':
            return self.f_totalConsCoal(demand)
        if is_fuel_type == 'is_oil':
            return self.f_totalConsOil(demand)
        if is_fuel_type == 'is_nuclear':
            return self.f_totalConsNuclear(demand)
        if is_fuel_type == 'is_hydro':
            return self.f_totalConsHydro(demand)
        if is_fuel_type == 'is_geothermal':
            return self.f_totalConsGeothermal(demand)
        if is_fuel_type == 'is_biomass':
            return self.f_totalConsBiomass(demand)


    def calcFullMeritOrder(self):
        """ Calculates the base_ and marg_ co2, so2, nox, and coal_mix, where "base_" represents the online "base load" that does not change with marginal changes in demand and "marg_" represents the marginal portion of the merit order that does change with marginal changes in demand. The calculation of base_ and marg_ changes depending on whether the minimum output constraint (the include_min_output variable) is being used. In general, "base" is a value (e.g. 'full_gen_cost_tot_base' has units [$], and 'full_co2_base' has units [kg]) while "marg" is a rate (e.g. 'full_gen_cost_tot_marg' has units [$/MWh], and 'full_co2_marg' has units [kg/MWh]). When the dispatch object solves the dispatch, it calculates the total emissions for one time period as 'full_co2_base' + 'full_co2_marg' * (marginal generation MWh) to end up with units of [kg].
        ---
        """

        df = self.df.copy(deep=True)
        binary_demand_is_below_demand_threshold = (scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal'))) > 0).values.astype(int)
        weight_marginal_unit = (1-self.mdt_weight) + self.mdt_weight*(1-binary_demand_is_below_demand_threshold)
        weight_mindowntime_units = 1 - weight_marginal_unit
        #INCLUDING MIN OUTPUT
        if self.include_min_output:

            #total production cost
            df['full_gen_cost_tot_base'] = 0.1*df.a.apply(self.returnTotalCost) + 0.9*df.s.apply(self.returnTotalCost) + df.s.apply(self.returnMarginalGenerator, args=('gen_cost',)) * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base production cost [$]
            df['full_gen_cost_tot_marg'] = ((df.s.apply(self.returnTotalCost) - df.a.apply(self.returnTotalCost)) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=('gen_cost',)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0) #calculate the marginal base production cost [$/MWh]

            #emissions
            for e in ['co2', 'so2', 'nox']:
                df['full_' + e + '_base'] = 0.1*df.a.apply(self.returnTotalEmissions, args=(e,)) + 0.9*df.s.apply(self.returnTotalEmissions, args=(e,)) + df.s.apply(self.returnMarginalGenerator, args=(e,)) * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base emissions [kg]
                #scipy.multiply(MEF of normal generation, weight of normal genearation) + scipy.multiply(MEF of mdt_reserves, weight of mdt_reserves) where MEF of normal generation is the calculation that happens without accounting for mdt, weight of normal generation is ((f-s) / ((f-s)) + mdt_reserves) and MEF of mdt_reserves is total_value_mdt_emissions / total_mw_mdt_reserves
                df['full_' + e + '_marg'] = scipy.multiply(((df.s.apply(self.returnTotalEmissions, args=(e,)) - df.a.apply(self.returnTotalEmissions, args=(e,))) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=(e,)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0)  ,    weight_marginal_unit  ) + scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalEmissions_Coal, args=(e,)) - self.returnTotalEmissions_Coal(self.coal_mdt_demand_threshold, e)))  ,  scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0)  ,  weight_mindowntime_units  )

            print((df.min_out/(df.f-df.s)))
            #emissions damages
            df['full_dmg_easiur_base'] = 0.1*df.a.apply(self.returnTotalEasiurDamages) + 0.9*df.s.apply(self.returnTotalEasiurDamages) + df.s.apply(self.returnMarginalGenerator, args=('dmg_easiur',)) * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base easiur damages [$]
            #scipy.multiply(MEF of normal generation, weight of normal genearation) + scipy.multiply(MEF of mdt_reserves, weight of mdt_reserves) where MEF of normal generation is the calculation that happens without accounting for mdt, weight of normal generation is ((f-s) / ((f-s)) + mdt_reserves) and MEF of mdt_reserves is total_value_mdt_emissions / total_mw_mdt_reserves
            df['full_dmg_easiur_marg'] = scipy.multiply(  ((df.s.apply(self.returnTotalEasiurDamages) - df.a.apply(self.returnTotalEasiurDamages)) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=('dmg_easiur',)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0)  ,  weight_marginal_unit  ) + scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalEasiurDamages_Coal) - self.returnTotalEasiurDamages_Coal(self.coal_mdt_demand_threshold)))  ,  scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0)  ,  weight_mindowntime_units  )

            #fuel mix
            for fl in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
            #for fl in ['gas', 'coal', 'oil']:
                df['full_' + fl + '_mix_base'] = 0.1*df.a.apply(self.returnTotalFuelMix, args=(('is_'+fl),)) + 0.9*df.s.apply(self.returnTotalFuelMix, args=(('is_'+fl),)) + self.df['is_'+fl] * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base coal_mix [MWh]
                #scipy.multiply(dmgs of normal generation, weight of normal genearation) + scipy.multiply(dmgs of mdt_reserves, weight of mdt_reserves) where dmgs of normal generation is the calculation that happens without accounting for mdt, weight of normal generation is ((f-s) / ((f-s)) + mdt_reserves) and dmgs of mdt_reserves is total_value_mdt_reserves / total_mw_mdt_reserves
                fuel_multiplier = scipy.where(fl=='coal', 1.0, 0.0)
                df['full_' + fl + '_mix_marg'] = scipy.multiply(  ((df.s.apply(self.returnTotalFuelMix, args=(('is_'+fl),)) - df.a.apply(self.returnTotalFuelMix, args=(('is_'+fl),))) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=(('is_'+fl),)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0)  ,  weight_marginal_unit  )  +  scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal'))), scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0) * fuel_multiplier  ,  weight_mindowntime_units  )

            #fuel consumption
            for fl in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
                df['full_' + fl + '_consumption_base'] = 0.1*df.a.apply(self.returnTotalFuelConsumption, args=(('is_'+fl),)) + 0.9*df.s.apply(self.returnTotalFuelConsumption, args=(('is_'+fl),)) + self.df['is_'+fl] * df.s.apply(self.returnMarginalGenerator, args=('heat_rate',)) * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base fuel consumption [mmBtu]
                #scipy.multiply(mmbtu/mw of normal generation, weight of normal genearation) + scipy.multiply(mmbtu/mw of mdt_reserves, weight of mdt_reserves) where mmbtu/mw of normal generation is the calculation that happens without accounting for mdt, weight of normal generation is ((f-s) / ((f-s)) + mdt_reserves) and mmbtu/mw of mdt_reserves is total_value_mdt_reserves / total_mw_mdt_reserves
                fuel_multiplier = scipy.where(fl=='coal', 1.0, 0.0)
                df['full_' + fl + '_consumption_marg'] = scipy.multiply(  ((df.s.apply(self.returnTotalFuelConsumption, args=('is_'+fl,)) - df.a.apply(self.returnTotalFuelConsumption, args=('is_'+fl,))) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=('is_'+fl,)) * df.s.apply(self.returnMarginalGenerator, args=('heat_rate',)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0)  ,  weight_marginal_unit  )  +  scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalFuelConsumption, args=(('is_coal'),)) - self.returnTotalFuelConsumption(self.coal_mdt_demand_threshold, 'is_coal'))), scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0) * fuel_multiplier ,  weight_mindowntime_units  )

        #EXCLUDING MIN OUTPUT
        if not self.include_min_output:
            #total production cost
            df['full_gen_cost_tot_base'] = df.s.apply(self.returnTotalCost) #calculate the base production cost, which is now the full load production cost of the generators in the merit order below the marginal unit [$]
            df['full_gen_cost_tot_marg'] = df.s.apply(self.returnMarginalGenerator, args=('gen_cost',)) #calculate the marginal production cost, which is now just the generation cost of the marginal generator [$/MWh]
            #emissions
            for e in ['co2', 'so2', 'nox']:
                df['full_' + e + '_base'] = df.s.apply(self.returnTotalEmissions, args=(e,)) #calculate the base emissions, which is now the full load emissions of the generators in the merit order below the marginal unit [kg]
                df['full_' + e + '_marg'] = scipy.multiply(  df.s.apply(self.returnMarginalGenerator, args=(e,))  ,   weight_marginal_unit  ) + scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalEmissions_Coal, args=(e,)) - self.returnTotalEmissions_Coal(self.coal_mdt_demand_threshold, e)))  ,  scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0)  ,  weight_mindowntime_units  )
            #emissions damages
            df['full_dmg_easiur_base'] = df.s.apply(self.returnTotalEasiurDamages) #calculate the total Easiur damages
            df['full_dmg_easiur_marg'] = scipy.multiply(  df.s.apply(self.returnMarginalGenerator, args=('dmg_easiur',))  ,  weight_marginal_unit  ) + scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalEasiurDamages_Coal) - self.returnTotalEasiurDamages_Coal(self.coal_mdt_demand_threshold)))  ,  scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0)  ,  weight_mindowntime_units  )
            #fuel mix
            for fl in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
                df['full_' + fl + '_mix_base'] = df.s.apply(self.returnTotalFuelMix, args=(('is_'+fl),)) #calculate the base fuel_mix, which is now the full load coal mix of the generators in the merit order below the marginal unit [MWh]
                fuel_multiplier = scipy.where(fl=='coal', 1.0, 0.0)
                df['full_' + fl + '_mix_marg'] = scipy.multiply(  df.s.apply(self.returnMarginalGenerator, args=(('is_'+fl),))  ,  weight_marginal_unit  )  +  scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal'))), scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0) * fuel_multiplier  ,  weight_mindowntime_units  )
            #fuel consumption
            for fl in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
                df['full_' + fl + '_consumption_base'] = df.s.apply(self.returnTotalFuelConsumption, args=(('is_'+fl),)) #calculate the base fuel_consumption, which is now the fuel consumption of the generators in the merit order below the marginal unit [MWh]
                fuel_multiplier = scipy.where(fl=='coal', 1.0, 0.0)
                df['full_' + fl + '_consumption_marg'] = scipy.multiply(  df.s.apply(self.returnMarginalGenerator, args=('is_'+fl,)) * df.s.apply(self.returnMarginalGenerator, args=('heat_rate',))  ,  weight_marginal_unit  )  +  scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalFuelConsumption, args=(('is_coal'),)) - self.returnTotalFuelConsumption(self.coal_mdt_demand_threshold, 'is_coal'))), scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0) * fuel_multiplier ,  weight_mindowntime_units  )
        #update the master dataframe df
        self.df = df


    def returnFullMarginalValue(self, demand, col_type):
        """ Given demand and col_type inputs, return the col_type (i.e. 'co2' for marginal co2 emissions rate or 'coal_mix' for coal share of the generation) of the marginal units in the Full model (the Full model includes the minimum output constraint).
        ---
        demand : [MW]
        col_type : 'co2', 'so2', 'nox', 'coal_mix', etc.
        return : full_"emissions_type"_marg as calculated in the Full model: calcFullMeritOrder
        """
        return self.returnMarginalGenerator(demand, 'full_' + col_type + '_marg')


    def createTotalInterpolationFunctionsFull(self):
        """ Creates interpolation functions for the full total data (i.e. total cost, total emissions, etc.) depending on total demand.
        """
        test = self.df.copy()
        #cost
        self.f_totalCostFull = scipy.interpolate.interp1d(test.demand, test['full_gen_cost_tot_base'] + (test['demand'] - test['s']) * test['full_gen_cost_tot_marg'])
        #emissions and health damages
        self.f_totalCO2Full = scipy.interpolate.interp1d(test.demand, test['full_co2_base'] + (test['demand'] - test['s']) * test['full_co2_marg'])
        self.f_totalSO2Full = scipy.interpolate.interp1d(test.demand, test['full_so2_base'] + (test['demand'] - test['s']) * test['full_so2_marg'])
        self.f_totalNOXFull = scipy.interpolate.interp1d(test.demand, test['full_nox_base'] + (test['demand'] - test['s']) * test['full_nox_marg'])
        self.f_totalDmgFull = scipy.interpolate.interp1d(test.demand, test['full_dmg_easiur_base'] + (test['demand'] - test['s']) * test['full_dmg_easiur_marg'])
        #fuel mix
        self.f_totalGasFull = scipy.interpolate.interp1d(test.demand, test['full_gas_mix_base'] + (test['demand'] - test['s']) * test['full_gas_mix_marg'])
        self.f_totalCoalFull = scipy.interpolate.interp1d(test.demand, test['full_coal_mix_base'] + (test['demand'] - test['s']) * test['full_coal_mix_marg'])
        self.f_totalOilFull = scipy.interpolate.interp1d(test.demand, test['full_oil_mix_base'] + (test['demand'] - test['s']) * test['full_oil_mix_marg'])
        self.f_totalNuclearFull = scipy.interpolate.interp1d(test.demand, test['full_nuclear_mix_base'] + (test['demand'] - test['s']) * test['full_nuclear_mix_marg'])
        self.f_totalHydroFull = scipy.interpolate.interp1d(test.demand, test['full_hydro_mix_base'] + (test['demand'] - test['s']) * test['full_hydro_mix_marg'])
        self.f_totalGeothermalFull = scipy.interpolate.interp1d(test.demand, test['full_geothermal_mix_base'] + (test['demand'] - test['s']) * test['full_geothermal_mix_marg'])
        self.f_totalBiomassFull = scipy.interpolate.interp1d(test.demand, test['full_biomass_mix_base'] + (test['demand'] - test['s']) * test['full_biomass_mix_marg'])
        #fuel consumption
        self.f_totalConsGasFull = scipy.interpolate.interp1d(test.demand, test['full_gas_consumption_base'] + (test['demand'] - test['s']) * test['full_gas_consumption_marg'])
        self.f_totalConsCoalFull = scipy.interpolate.interp1d(test.demand, test['full_coal_consumption_base'] + (test['demand'] - test['s']) * test['full_coal_consumption_marg'])
        self.f_totalConsOilFull = scipy.interpolate.interp1d(test.demand, test['full_oil_consumption_base'] + (test['demand'] - test['s']) * test['full_oil_consumption_marg'])
        self.f_totalConsNuclearFull = scipy.interpolate.interp1d(test.demand, test['full_nuclear_consumption_base'] + (test['demand'] - test['s']) * test['full_nuclear_consumption_marg'])
        self.f_totalConsHydroFull = scipy.interpolate.interp1d(test.demand, test['full_hydro_consumption_base'] + (test['demand'] - test['s']) * test['full_hydro_consumption_marg'])
        self.f_totalConsGeothermalFull = scipy.interpolate.interp1d(test.demand, test['full_geothermal_consumption_base'] + (test['demand'] - test['s']) * test['full_geothermal_consumption_marg'])
        self.f_totalConsBiomassFull = scipy.interpolate.interp1d(test.demand, test['full_biomass_consumption_base'] + (test['demand'] - test['s']) * test['full_biomass_consumption_marg'])


    def returnFullTotalValue(self, demand, col_type):
        """ Given demand and col_type inputs, return the total column of the online power plants in the Full model (the Full model includes the minimum output constraint).
        ---
        demand : [MW]
        col_type : 'co2', 'so2', 'nox', 'coal_mix', etc.
        return : total emissions = base emissions (marginal unit) + marginal emissions (marginal unit) * (D - s)
        """
        if col_type == 'gen_cost_tot':
            return self.f_totalCostFull(demand)
        if col_type == 'co2':
            return self.f_totalCO2Full(demand)
        if col_type == 'so2':
            return self.f_totalSO2Full(demand)
        if col_type == 'nox':
            return self.f_totalNOXFull(demand)
        if col_type == 'dmg_easiur':
            return self.f_totalDmgFull(demand)
        if col_type == 'gas_mix':
            return self.f_totalGasFull(demand)
        if col_type == 'coal_mix':
            return self.f_totalCoalFull(demand)
        if col_type == 'oil_mix':
            return self.f_totalOilFull(demand)
        if col_type == 'nuclear_mix':
            return self.f_totalNuclearFull(demand)
        if col_type == 'hydro_mix':
            return self.f_totalHydroFull(demand)
        if col_type == 'geothermal_mix':
            return self.f_totalGeothermalFull(demand)
        if col_type == 'biomass_mix':
            return self.f_totalBiomassFull(demand)
        if col_type == 'gas_consumption':
            return self.f_totalConsGasFull(demand)
        if col_type == 'coal_consumption':
            return self.f_totalConsCoalFull(demand)
        if col_type == 'oil_consumption':
            return self.f_totalConsOilFull(demand)
        if col_type == 'nuclear_consumption':
            return self.f_totalConsNuclearFull(demand)
        if col_type == 'hydro_consumption':
            return self.f_totalConsHydroFull(demand)
        if col_type == 'geothermal_consumption':
            return self.f_totalConsGeothermalFull(demand)
        if col_type == 'biomass_consumption':
            return self.f_totalConsBiomassFull(demand)

class dispatch(object):
    def __init__(self, bid_stack_object, demand_df, time_array=0):
        """ Read in bid stack object and the demand data. Solve the dispatch by projecting the bid stack onto the demand time series, updating the bid stack object regularly according to the time_array
        ---
        gen_data_object : a object defined by class generatorData
        bid_stack_object : a bid stack object defined by class bidStack
        demand_df : a dataframe with the demand data
        time_array : a scipy array containing the time intervals that we are changing fuel price etc. for. E.g. if we are doing weeks, then time_array=scipy.arange(52) + 1 to get an array of (1, 2, 3, ..., 51, 52)
        """
        self.bs = bid_stack_object
        self.df = demand_df
        self.time_array = time_array
        self.addDFColumns()


    def addDFColumns(self):
        """ Add additional columns to self.df to hold the results of the dispatch. New cells initially filled with zeros
        ---
        """
        indx = self.df.index
        cols = scipy.array(('gen_cost_marg', 'gen_cost_tot', 'co2_marg', 'co2_tot', 'so2_marg', 'so2_tot', 'nox_marg', 'nox_tot', 'dmg_easiur', 'biomass_mix', 'coal_mix', 'gas_mix', 'geothermal_mix', 'hydro_mix', 'nuclear_mix', 'oil_mix', 'marg_gen', 'coal_mix_marg', 'marg_gen_fuel_type', 'mmbtu_coal', 'mmbtu_gas', 'mmbtu_oil'))
        dfExtension = pandas.DataFrame(index=indx, columns=cols).fillna(0)
        self.df = pandas.concat([self.df, dfExtension], axis=1)


    def calcDispatchSlice(self, bstack, start_date=0, end_date=0):
        """ For each datum in demand time series (e.g. each hour) between start_date and end_date calculate the dispatch
        ---
        bstack: an object created using the simple_dispatch.bidStack class
        start_datetime : string of format '2014-01-31' i.e. 'yyyy-mm-dd'. If argument == 0, uses start date of demand time series
        end_datetime : string of format '2014-01-31' i.e. 'yyyy-mm-dd'. If argument == 0, uses end date of demand time series
        """
        if start_date==0:
            start_date = self.df.datetime.min()
        else:
            start_date = pandas._libs.tslib.Timestamp(start_date)
        if end_date==0:
            end_date = self.df.datetime.max()
        else:
            end_date = pandas._libs.tslib.Timestamp(end_date)
        #slice of self.df within the desired dates
        df_slice = self.df[(self.df.datetime >= pandas._libs.tslib.Timestamp(start_date)) & (self.df.datetime < pandas._libs.tslib.Timestamp(end_date))].copy(deep=True)
        #calculate the dispatch for the slice by applying the return###### functions of the bstack object
        df_slice['gen_cost_marg'] = df_slice.demand.apply(bstack.returnMarginalGenerator, args=('gen_cost',)) #generation cost of the marginal generator ($/MWh)
        df_slice['gen_cost_tot'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('gen_cost_tot',)) #generation cost of the total generation fleet ($)
        for e in ['co2', 'so2', 'nox']:
            df_slice[e + '_marg'] = df_slice.demand.apply(bstack.returnFullMarginalValue, args=(e,)) #emissions rate (kg/MWh) of marginal generators
            df_slice[e + '_tot'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=(e,)) #total emissions (kg) of online generators
        df_slice['dmg_easiur'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('dmg_easiur',)) #total easiur damages ($)
        for f in ['gas', 'oil', 'coal', 'nuclear', 'biomass', 'geothermal', 'hydro']:
            df_slice[f + '_mix'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=(f+'_mix',))
        df_slice['coal_mix_marg'] = df_slice.demand.apply(bstack.returnFullMarginalValue, args=('coal_mix',))
        df_slice['marg_gen_fuel_type'] = df_slice.demand.apply(bstack.returnMarginalGenerator, args=('fuel_type',))
        df_slice['mmbtu_coal'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('coal_consumption',)) #total coal mmBtu
        df_slice['mmbtu_gas'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('gas_consumption',)) #total gas mmBtu
        df_slice['mmbtu_oil'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('oil_consumption',)) #total oil mmBtu
        self.df[(self.df.datetime >= pandas._libs.tslib.Timestamp(start_date)) & (self.df.datetime < pandas._libs.tslib.Timestamp(end_date))] = df_slice


    def createDfMdtCoal(self, demand_threshold, time_t):
        """ For a given demand threshold, creates a new version of the generator data that approximates the minimum down time constraint for coal plants
        ---
        demand_threshold: the system demand below which some coal plants will turn down to minimum rather than turning off
        returns a dataframe of the same format as gd.df but updated so the coal generators in the merit order below demand_threshold have their capacities reduced by their minimum output, their minimum output changed to zero, and the sum of their minimum outputs applied to the capacity of coal_0, where coal_0 also takes the weighted average of their heat rates, emissions, rates, etc. Note that this new dataframe only contains the updated coal plants, but not the complete gd.df information (i.e. for gas plants and higher cost coal plants), but it can be incorporated back into the original df (i.e. bs.df_0) using the pandas update command.
        """
        #set the t (time i.e. week) object
        t = time_t
        #get the orispl_unit information for the generators you need to adjust
        coal_mdt_orispl_unit_list = list(self.bs.df[(self.bs.df.fuel_type=='coal') & (self.bs.df.demand <= demand_threshold)].orispl_unit.copy().values)
        coal_mdt_gd_idx = self.bs.df_0[self.bs.df_0.orispl_unit.isin(coal_mdt_orispl_unit_list)].index

        #create a new set of generator data where there is a large coal unit at the very bottom representing the baseload of the coal generators if they do not turn down below their minimum output, and all of the coal generators have their capacity reduced to (1-min_output).
        df_mdt_coal = self.bs.df_0[self.bs.df_0.orispl_unit.isin(coal_mdt_orispl_unit_list)][['orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'vom', 'min_out_multiplier', 'min_out', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t, 'dmg%i'%t]].copy()
        df_mdt_coal = df_mdt_coal[df_mdt_coal.orispl_unit != 'coal_0']
        #create a pandas Series that will hold the large dummy coal unit that represents coal base load
        df_mdt_coal_base = df_mdt_coal.copy().iloc[0]
        df_mdt_coal_base[['orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'min_out_multiplier', 'min_out']] = ['coal_0', 'sub', 'coal', 'st', 0.0, 0.0]
        #columns for the week we are currently solving
        t_columns = ['orispl_unit', 'fuel_type', 'prime_mover', 'vom', 'min_out_multiplier', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t, 'dmg%i'%t]
        df_mdt_coal_base_temp = df_mdt_coal[t_columns].copy()
        #the capacity of the dummy coal unit will be the sum of the minimum output of all the coal units
        df_mdt_coal_base_temp[['mw%i'%t]] = df_mdt_coal_base_temp['mw%i'%t] * df_mdt_coal_base_temp.min_out_multiplier
        #the vom, co2, so2, nox, heat_rate, fuel_price, and dmg of the dummy unit will equal the weighted average of the other coal plants
        weighted_cols = df_mdt_coal_base_temp.columns.drop(['orispl_unit', 'fuel_type', 'prime_mover', 'min_out_multiplier', 'mw%i'%t])
        df_mdt_coal_base_temp[weighted_cols] = df_mdt_coal_base_temp[weighted_cols].multiply(df_mdt_coal_base_temp['mw%i'%t], axis='index') / df_mdt_coal_base_temp['mw%i'%t].sum()
        df_mdt_coal_base_temp = df_mdt_coal_base_temp.sum(axis=0)
        #update df_mdt_coal_base with df_mdt_coal_base_temp, which holds the weighted average characteristics of the other coal plants
        df_mdt_coal_base[['vom', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t, 'dmg%i'%t]] = df_mdt_coal_base_temp[['vom', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t, 'dmg%i'%t]]
        #reduce the capacity of the other coal plants by their minimum outputs (since their minimum outputs are now a part of coal_0)
        df_mdt_coal.loc[df_mdt_coal.fuel_type == 'coal','mw%i'%t] = df_mdt_coal[df_mdt_coal.fuel_type == 'coal'][['mw%i'%t]].multiply((1-df_mdt_coal[df_mdt_coal.fuel_type == 'coal'].min_out_multiplier), axis='index')
        #add coal_0 to df_mdt_coal
        df_mdt_coal = df_mdt_coal.append(df_mdt_coal_base, ignore_index = True)
        #change the minimum output of the coal plants to 0.0
        df_mdt_coal.loc[df_mdt_coal.fuel_type == 'coal',['min_out_multiplier', 'min_out']] = [0.0, 0.0]
        #update the index to match the original bidStack
        df_mdt_coal.index = coal_mdt_gd_idx
        return df_mdt_coal


    def calcMdtCoalEventsT(self, start_datetime, end_datetime, coal_merit_order_input_df):
        """ For a given demand threshold, creates a new version of the generator data that approximates the minimum down time constraint for coal plants
        ---
        demand_threshold: the system demand below which some coal plants will turn down to minimum rather than turning off
        returns a dataframe of the same format as gd.df but updated so the coal generators in the merit order below demand_threshold have their capacities reduced by their minimum output, their minimum output changed to zero, and the sum of their minimum outputs applied to the capacity of coal_0, where coal_0 also takes the weighted average of their heat rates, emissions, rates, etc.
        """
        #the function below returns the demand value of the merit_order_input_df that is just above the demand_input_scalar
        def bisect_column(demand_input_scalar, merit_order_input_df):
            try:
                out = coal_merit_order_input_df.iloc[bisect_left(list(coal_merit_order_input_df.demand),demand_input_scalar)].demand
        #if demand_threshold exceeds the highest coal_merit_order.demand value (i.e. all of min output constraints are binding for coal)
            except:
                out = coal_merit_order_input_df.iloc[-1].demand
            return out
        #bring in the coal mdt events calculated in generatorData
        mdt_coal_events_t = self.bs.mdt_coal_events.copy()
        #slice the coal mdt events based on the current start/end section of the dispatch solution
        mdt_coal_events_t = mdt_coal_events_t[(mdt_coal_events_t.end >= start_datetime) & (mdt_coal_events_t.start <= end_datetime)]
        #translate the demand_thresholds into the next highest demand data in the merit_order_input_df. This will allow us to reduce the number of bidStacks we need to generate. E.g. if two days have demand thresholds of 35200 and 35250 but the next highest demand in the coal merit order is 36000, then both of these days can use the 36000 mdt_bidStack, and we can recalculate the bidStack once instead of twice.
        mdt_coal_events_t[['demand_threshold']] = mdt_coal_events_t.demand_threshold.apply(bisect_column, args=(coal_merit_order_input_df,))
        return mdt_coal_events_t


    def calcDispatchAll(self):
        """ Runs calcDispatchSlice for each time slice in the fuel_prices_over_time dataframe, creating a new bidstack each time. So, fuel_prices_over_time contains multipliers (e.g. 0.95 or 1.14) for each fuel type (e.g. ng, lig, nuc) for different slices of time (e.g. start_date = '2014-01-07' and end_date = '2014-01-14'). We use these multipliers to change the fuel prices seen by each generator in the bidStack object. After changing each generator's fuel prices (using bidStack.updateFuelPrices), we re-calculate the bidStack merit order (using bidStack.calcGenCost), and then calculate the dispatch for the slice of time defined by the fuel price multipliers. This way, instead of calculating the dispatch over the whole year, we can calculate it in chunks of time (e.g. weeks) where each chunk of time has different fuel prices for the generators.
        Right now the only thing changing per chunk of time is the fuel prices based on trends in national commodity prices. Future versions might try and do regional price trends and add things like maintenance downtime or other seasonal factors.
        ---
        fills in the self.df dataframe one time slice at a time
        """
        #run the whole solution if self.fuel_prices_over_time isn't being used
        if scipy.shape(self.time_array) == (): #might be a more robust way to do this. Would like to say if ### == 0, but doing that when ### is a dataframe gives an error
            self.calcDispatchSlice(self.bs)
        #otherwise, run the dispatch in time slices, updating the bid stack each slice
        else:
            for t in self.time_array:
                print (str(round(t/float(len(self.time_array)),3)*100) + '% Complete')
                #update the bidStack object to the current week
                self.bs.updateTime(t)
                #calculate the dispatch for the time slice over which the updated fuel prices are relevant
                start = (datetime.datetime.strptime(str(self.bs.year) + '-01-01', '%Y-%m-%d') + datetime.timedelta(days=7.05*(t-1)-1)).strftime('%Y-%m-%d')
                end = (datetime.datetime.strptime(str(self.bs.year) + '-01-01', '%Y-%m-%d') + datetime.timedelta(days=7.05*(t)-1)).strftime('%Y-%m-%d')
                #note that calcDispatchSlice updates self.df, so there is no need to do it in this calcDispatchAll function
                self.calcDispatchSlice(self.bs, start_date=start ,end_date=end)
                #coal minimum downtime
                #recalculate the dispatch for times when generatorData pre-processing estimates that the minimum downtime constraint for coal plants would trigger
                #define the coal merit order
                coal_merit_order = self.bs.df[(self.bs.df.fuel_type == 'coal')][['orispl_unit', 'demand']]
                #slice and bin the coal minimum downtime events
                events_mdt_coal_t = self.calcMdtCoalEventsT(start, end, coal_merit_order)
                #create a dictionary for holding the updated bidStacks, which change depending on the demand_threshold
                bs_mdt_dict = {}
                #for each unique demand_threshold
                for dt in events_mdt_coal_t.demand_threshold.unique():
                    #create an updated version of gd.df
                    gd_df_mdt_temp = self.bs.df_0.copy()
                    gd_df_mdt_temp.update(self.createDfMdtCoal(dt, t))
                    #use that updated gd.df to create an updated bidStack object, and store it in the bs_mdt_dict
                    bs_temp = copy.deepcopy(bs)
                    bs_temp.coal_mdt_demand_threshold = dt
                    bs_temp.updateDf(gd_df_mdt_temp)
                    bs_mdt_dict.update({dt:bs_temp})
                #for each minimum downtime event, recalculate the dispatch by inputting the bs_mdt_dict bidStacks into calcDispatchSlice to override the existing dp.df results datafram
                for i, e in events_mdt_coal_t.iterrows():
                    self.calcDispatchSlice(bs_mdt_dict[e.demand_threshold], start_date=e.start ,end_date=e.end)


if __name__ == '__main__':
    run_year = 2016
    nerc_region = 'PJM'
    pjm_dispatch_save_folder = 'pjm/'
    simulated_dispatch_save_folder = 'simulated/'

    hist_dispatch = pandas.read_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + '_hourly_demand_and_fuelmix.csv')
    demand_data = pandas.read_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + 'demand_data.csv')
    demand_data['datetime']= pandas.to_datetime(demand_data['datetime'])
    mdt_coal_events = pandas.read_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + 'mdt_coal_events.csv')
    df = pandas.read_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + 'df.csv')

    gd_short = {'year': run_year, 'nerc': 'PJM', 'hist_dispatch': hist_dispatch, 'demand_data': demand_data, 'mdt_coal_events': mdt_coal_events, 'df': df}
    
    for nr in [0]:
    #for nr in [0, 4, 10, 25, 50, 100, 200]: #base case
    #for nr in [2, 6, 8, 15, 20, 30, 40, 60, 80, 125, 150, 1000]: #base case
    #for nr in [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 1000]:
        co2_dol_per_ton = nr
        #run the bidStack object - use information about the generators (from gd) to create a merit order (bid stack) of the nerc region's generators
        bs = bidStack(gd_short, co2_dol_per_kg=(co2_dol_per_ton / 907.185), time=30, dropNucHydroGeo=True, include_min_output=False, mdt_weight=0.5) #NOTE: set dropNucHydroGeo to True if working with data that only looks at fossil fuels (e.g. CEMS)

        #run the dispatch object - use the nerc region's merit order (bs), a demand timeseries (gd.demand_data), and a time array (default is array([ 1,  2, ... , 51, 52]) for 52 weeks to run a whole year)
        bs.df.to_csv(pjm_dispatch_save_folder + str(run_year) + '_' + nerc_region + '_bidstack_df.csv')

        dp = dispatch(bs, demand_data, time_array = 52) #set up the object
        dp.calcDispatchAll() #function that solves the dispatch for each time period in time_array (default for each week of the year)
        dp.df.to_csv(simulated_dispatch_save_folder + 'dispatch_output_weekly_%s_%s_with_52.csv'%(nerc_region, str(run_year)), index=False)
