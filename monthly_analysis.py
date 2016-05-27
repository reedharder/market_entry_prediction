# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:33:10 2016

@author: Reed
"""
import os
import numpy as np
import pandas as pd
from itertools import product, combinations
import ast
import pickle
from sklearn.metrics import confusion_matrix
import  matplotlib.pyplot as plt
import itertools
import scipy.io as sio
import time
import networkx as nx
from  sklearn import linear_model
try: 
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
except NameError:
    os.chdir("C:/Users/reed/desktop/m106jproj/")


node_sets = {'ope35_sansHLN' : ['ATL', 'BOS', 'BWI', 'CLE', 'CLT', 'CVG', 'DCA', 'DEN', 'DFW', 'DTW', 'EWR', 'FLL', 'IAD', 'IAH', 'JFK', 'LAS', 'LAX', 'LGA', 'MCO', 'MDW', 'MEM', 'MIA', 'MSP', 'ORD', 'PDX', 'PHL', 'PHX', 'PIT', 'SAN', 'SEA', 'SFO', 'SLC', 'STL', 'TPA'],
'western': ['SEA','PDX','SFO','SAN','LAX','LAS','PHX','OAK','ONT','SMF','SJC'],
'top100_2014':['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'IAH', 'SFO', 'PHX', 'LAS', 'SEA', 'MSP', 'BOS', 'SLC', 'EWR', 'MCO', 'DTW', 'LGA', 'CLT', 'JFK', 'MDW', 'BWI', 'SAN', 'MIA', 'PHL', 'DCA', 'TPA', 'HOU', 'FLL', 'IAD', 'PDX', 'STL', 'BNA', 'HNL', 'MCI','OAK', 'DAL', 'AUS', 'SJC', 'SMF', 'RDU', 'SNA', 'MSY', 'MKE', 'SAT', 'CLE', 'IND', 'PIT', 'SJU', 'ABQ', 'CMH', 'OGG', 'OKC', 'BDL', 'ANC', 'BUR', 'JAX', 'CVG', 'ONT', 'OMA', 'TUL', 'RIC', 'ELP', 'RSW', 'BUF', 'RNO', 'PBI', 'CHS', 'LIT', 'TUS', 'MEM','SDF', 'BHM', 'LGB', 'PVD', 'KOA', 'BOI', 'GRR', 'LIH', 'ORF', 'FAT', 'XNA', 'DAY', 'MAF', 'GEG', 'MSN', 'DSM', 'COS', 'GSO', 'TYS', 'ALB', 'SAV', 'PNS', 'BTR', 'ICT', 'ROC', 'JAN', 'MHT', 'AMA','FSD', 'HPN']}


def nonstop_market_profile_monthly(output_file = "market_profiles/monthly_market_profile_%sm%s.csv",year = 2007, months=range(1,13), \
    t100_fn="bts_data/T100_2007.csv",p52_fn="bts_data/SCHEDULE_P52_2007.csv", t100_avgd_fn="processed_data/t100_avgd_m%s.csv", merge_HP=True, \
    t100_summed_fn = 'processed_data/t100_summed_m%s.csv', t100_craft_avg_fn='processed_data/t100_craft_avg_m%s.csv',\
    ignore_mkts = [], craft_freq_cuttoff = .01,max_competitors=100,\
    freq_cuttoff = .5, ms_cuttoff=.05, fs_cuttoff = .05, only_big_carriers=False, airports = node_sets['top100_2014']):
        
    #dictionary of month to quarter
    month_to_q = {1:1,2:1,3:1,4:2,5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4}
    common_year_days_month = {1 :31,
    2:	28,
    3:	31,
    4:	30,
    5:	31,
    6:	30,
    7:	31,
    8:	31,
    9:	30,
    10:	31,
    11:	30,
    12:	31}
    
    leap_year_days_month = {1 :31,
    2:	29,
    3:	31,
    4:	30,
    5:	31,
    6:	30,
    7:	31,
    8:	31,
    9:	30,
    10:	31,
    11:	30,
    12:	31}
    
    # is this a leap year?
    if year % 4 !=0:
        leap =False
    elif year % 100 !=0:
        leap=True
    elif year % 400 !=0:
        leap =False
    else: 
        leap = True
    # select day to month dict
    days_month_dict = leap_year_days_month if leap else common_year_days_month
        
    #read in revelant bts files and supplementary data files 
    
    ##os.chdir(directory)
    t100 = pd.read_csv(t100_fn)
    p52 = pd.read_csv(p52_fn)

    #create bidrectional market pairs
    pairs =[sorted([pair[0],pair[1]]) for pair in product(airports,airports) if pair[0]!=pair[1] ]
    txtpairs = list(set(["_".join(pair) for pair in pairs]))
    txtpairs = [pair for pair in txtpairs if pair not in ignore_mkts]
    
    #leave out fare finding for now, may add later
    #get relevant segments within network for all market pairs
    print("creating markets...")
    t100['BI_MARKET']=t100.apply(create_market,1) #first, create bidriectional market indicator   
    print("done")
    relevant_t100= t100.set_index('BI_MARKET').loc[txtpairs].reset_index() #then, select markets
    
    relevant_t100['BI_MARKET']=relevant_t100['index']
    #merge carrier HP under UA if this is called for.
    if merge_HP:
        relevant_t100['UNIQUE_CARRIER']=relevant_t100['UNIQUE_CARRIER'].replace('HP','US')

    #get relevant data from schedule P-5.2 (financials)
    relevant_p52_d = p52[p52['REGION']=='D']
    ##relevant_p52=relevant_p52_d[relevant_p52_d['QUARTER']==month_to_q[month]]
    
    #average quarterly costs if necessary 
    expenses_by_type = relevant_p52_d[['QUARTER','AIRCRAFT_TYPE','UNIQUE_CARRIER','TOT_AIR_OP_EXPENSES', 'TOTAL_AIR_HOURS']].dropna()   
    #calculate expenses per air hour for each type for each airline
    expenses_by_type['EXP_PER_HOUR'] = expenses_by_type['TOT_AIR_OP_EXPENSES'] / expenses_by_type['TOTAL_AIR_HOURS']

    #average relevant monthly frequency to get daily freqencies
    t100fields =['MONTH', 'QUARTER','BI_MARKET','UNIQUE_CARRIER','ORIGIN', 'DEST','AIRCRAFT_TYPE','DEPARTURES_SCHEDULED','DEPARTURES_PERFORMED','SEATS','PASSENGERS','DISTANCE','AIR_TIME']
    #monthly departures, daily seats, daily passengers, avg distance, total airtime, aggregated across months
    t100_summed = relevant_t100[relevant_t100['MONTH'].apply( lambda x: not np.isnan(x))]
    
    #convert airtime to hours
    t100_summed['AIR_HOURS']=(t100_summed['AIR_TIME']/60)
    t100_summed['FLIGHT_TIME']=t100_summed['AIR_HOURS']/t100_summed['DEPARTURES_PERFORMED']
    #get frequency per day
    t100_summed['DAILY_FREQ']=t100_summed['DEPARTURES_SCHEDULED']/t100_summed['MONTH'].apply(lambda row: days_month_dict[int(row)]) 
    #get average number available seats per flight
    t100_summed['SEATS_PER_FLIGHT'] = t100_summed['SEATS']/t100_summed['DEPARTURES_PERFORMED']  #CHECK NUMBERS
    
    #MAKE SURE MONTHS PRESERVED BELOW
    
    
    #drop unnessessary total airtime column
    t100_summed = t100_summed.drop('AIR_TIME',axis=1)
    t0 = time.time()
    t100fields =['QUARTER','MONTH','BI_MARKET','UNIQUE_CARRIER','AIRCRAFT_TYPE','DEPARTURES_SCHEDULED','SEATS','SEATS_PER_FLIGHT','PASSENGERS','DISTANCE','AIR_HOURS', 'DAILY_FREQ']
    #merge t100 data with cost data
    t100_summed=pd.merge(t100_summed,expenses_by_type,on=['AIRCRAFT_TYPE','UNIQUE_CARRIER','QUARTER'])
    #Calculate cost per  invididual flight for each flight type (quarterly expenses over departures performed per quarter)
    t100_summed['FLIGHT_COST'] = t100_summed['AIR_HOURS']*t100_summed['EXP_PER_HOUR']/t100_summed['DEPARTURES_PERFORMED'] #get cost per flight type
    #filter empty flights
    t100_summed = t100_summed[t100_summed['PASSENGERS']>0]
    t100_summed = t100_summed[t100_summed['DEPARTURES_SCHEDULED']>0] 
    #additional filters: frequency, unrealistic seats per flight 
    t100_summed = t100_summed[t100_summed['DAILY_FREQ']>=craft_freq_cuttoff]
    #check for extreme seat numbers
    if t100_summed[(t100_summed['SEATS_PER_FLIGHT']<10) | (t100_summed['SEATS_PER_FLIGHT']>500)].shape[0]>0:
        print("Average seat anomalies:")
        print(t100_summed[(t100_summed['SEATS_PER_FLIGHT']<10) | (t100_summed['SEATS_PER_FLIGHT']>500)])
    #fix here 
    t100_summed.to_csv(t100_summed_fn % year)
    t100fields =['MONTH','QUARTER','BI_MARKET','ORIGIN','DEST','UNIQUE_CARRIER','AIRCRAFT_TYPE','DEPARTURES_SCHEDULED','SEATS','SEATS_PER_FLIGHT','PASSENGERS','DISTANCE', 'DAILY_FREQ','FLIGHT_COST','FLIGHT_TIME','AIR_HOURS']
    # calculate average cost, airtime, and seats per flight, per carrier per directional segment across craft types, weighted by frequency of constituent aircraft types
    t100_summed_avgs = t100_summed[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET','ORIGIN','DEST','QUARTER','MONTH']).apply(craft_weight_avgs)
    # calculate averages over craft 
    #REASON WHY THIS WORKED IN PIPELINE?
    t100_craft_avg = t100_summed_avgs[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET','ORIGIN','DEST','QUARTER','MONTH']).aggregate({'DEPARTURES_SCHEDULED':np.sum,'SEATS':np.sum,'SEATS_PER_FLIGHT':np.mean,'PASSENGERS':np.sum,'DISTANCE':np.mean, 'DAILY_FREQ':np.sum,'FLIGHT_COST':np.mean,'FLIGHT_TIME':np.mean,'AIR_HOURS':np.sum}).reset_index()  
    #save file of t100 summed over months and averaged over craft, to check passenger equivalence between market directions
    t100_craft_avg.to_csv(t100_craft_avg_fn % (year)) #USE THIS FOR ADJACENY. CHECK FOR BAKFORTH DISCORDANCE
    
    
    #average values between segments sharing a bidirectional market 
    t100fields =['QUARTER','MONTH','BI_MARKET','UNIQUE_CARRIER','DEPARTURES_SCHEDULED','SEATS','SEATS_PER_FLIGHT','PASSENGERS','DISTANCE', 'DAILY_FREQ','FLIGHT_COST','FLIGHT_TIME','AIR_HOURS']
    t100_avgd = t100_craft_avg[t100fields].groupby(['UNIQUE_CARRIER','BI_MARKET','QUARTER','MONTH']).aggregate({'DEPARTURES_SCHEDULED':np.mean,'DAILY_FREQ':np.mean,'SEATS':np.mean,'PASSENGERS':np.mean,'DISTANCE':np.mean,'FLIGHT_COST': np.mean,'SEATS_PER_FLIGHT':np.mean,'FLIGHT_TIME':np.mean,'AIR_HOURS':np.mean}).reset_index()
    #save data frame to csv: costs and frequencies by market, carrier, aircraft type
    t100_avgd.to_csv(t100_avgd_fn % (year),sep="\t")  
    #remove entries below daily frequency cuttoff
    t100_avgd_clip = t100_avgd[t100_avgd['DAILY_FREQ']>=freq_cuttoff]
    
    t1 = time.time()-t0    
    
    
    #group and rank carriers within markets, calculate market shares, market totals, etc
    t100_grouped = t100_avgd_clip.groupby(['MONTH','BI_MARKET'])
    for month in months:
        grouplist = []
        for market in list(set(t100_avgd_clip[t100_avgd_clip['MONTH']==month]['BI_MARKET'].tolist())):
            market_group = t100_grouped.get_group((month,market))
            new_group = market_rank(market_group, ms_cuttoff=ms_cuttoff,fs_cuttoff=fs_cuttoff)
            grouplist.append(new_group)
        t100ranked = pd.concat(grouplist,axis=0)
        t100ranked=t100ranked.sort(columns=['BI_MARKET','MARKET_RANK'])            
        #remove markets more with more than max competitors from model   
        original_count = t100ranked.shape[0]
        t100ranked = t100ranked[t100ranked['MARKET_COMPETITORS']<=max_competitors]
        new_count = t100ranked.shape[0]
        print('removed %s carrier-segments with more than max competitors, out of %s in total' % (original_count - new_count, original_count))    
        #save t100ranked to file
        t100ranked.to_csv(output_file % (year, month))    
        print([year,month])
    return t100ranked
'''
helper function to create a bidirectional market indicator (with airports sorted by text) for origin-destination pairs
'''    
def create_market(row):
    market = [row['ORIGIN'], row['DEST']]
    market.sort()
    return "_".join(market)
    


'''
helper function get a weighed average costs and flight times across a directional market
'''
def craft_weight_avgs(gb):
    cost_weighted = np.average(gb['FLIGHT_COST'], weights=gb['DAILY_FREQ'])
    gb['FLIGHT_COST'] = np.repeat(cost_weighted,gb.shape[0])
    time_weighted = np.average(gb['FLIGHT_TIME'], weights=gb['DAILY_FREQ'])
    gb['FLIGHT_TIME'] = np.repeat(time_weighted,gb.shape[0])
    seats_weighted = np.average(gb['SEATS_PER_FLIGHT'], weights=gb['DAILY_FREQ'])
    gb['SEATS_PER_FLIGHT'] = np.repeat(seats_weighted,gb.shape[0])            
    return gb
'''
helper function to average across aircraft types and rank carriers by passenger flow 
via pandas groupby function, recieves sub-dataframes, each one comprising a market
'''      
def market_rank(gb, ms_cuttoff,fs_cuttoff):                                  
    Mtot = gb['PASSENGERS'].sum()
    Ftot =gb['DAILY_FREQ'].sum()
    gb['FREQ_TOT'] = np.repeat(Ftot,gb.shape[0] )   
    gb['MARKET_TOT'] = np.repeat(Mtot,gb.shape[0] )    
    Mcount =gb.shape[0]
    gb['MARKET_COMPETITORS'] = np.repeat(Mcount,gb.shape[0] )
    rank = np.array(gb['PASSENGERS'].tolist()).argsort()[::-1].argsort() +1 
    gb['MARKET_RANK'] = rank         
    gb = gb.sort(columns=['MARKET_RANK'],ascending=True,axis =0)        
    gb['MS_TOT']=gb['PASSENGERS']/gb['MARKET_TOT']
    gb['FS_TOT']=gb['DAILY_FREQ']/gb['FREQ_TOT']
    #cumulative market share upto and including that ranking
    gb['CUM_MS']=gb.apply(lambda x: gb['MS_TOT'][:x['MARKET_RANK']].sum(), axis=1)
    #cumulative market share upto that ranking
    gb['PREV_CUM_MS']=gb.apply(lambda x: gb['MS_TOT'][:x['MARKET_RANK']-1].sum(), axis=1)
    #remove those carriers that appear after cuttoff
    gb=gb[gb['MS_TOT']>=ms_cuttoff]
    gb=gb[gb['FS_TOT']>=fs_cuttoff]
    #recalculate market shares
    Mtot = gb['PASSENGERS'].sum()
    Ftot =gb['DAILY_FREQ'].sum()
    #get total market size
    gb['MARKET_TOT'] = np.repeat(Mtot,gb.shape[0] )   
    gb['FREQ_TOT'] = np.repeat(Ftot,gb.shape[0] )  
    #get total number of competitors in market and save as column 
    Mcount =gb.shape[0]
    gb['MARKET_COMPETITORS'] = np.repeat(Mcount,gb.shape[0] )
    #get market share as passengers for that carrier over total market size 
    gb['MS_TOT']=gb['PASSENGERS']/gb['MARKET_TOT']
    gb['FS_TOT']=gb['DAILY_FREQ']/gb['FREQ_TOT']
    return gb 
    
'''
function to create adjaceny matrices of airline freqencies and costs (where airline present in market), for each airline at each time step (month)
### REPRESENTS COSTS, WHICH ARE PROBLEMATIC - > PERHAPS REMAKE T100 TO ACCOUNT FOR STRANGE BEHAVIOR
'''
def compile_adjaceny_mats():
    profile_file = "market_profiles/monthly_market_profile_%sm%s.csv"
    airports = sorted(node_sets['top100_2014'])
    years = list(range(2007,2016))
    months = list(range(1,13))
    profile_list = []
    # load all market profile files
    time_step_count = 0 #keep track of numer of time steps
    time_step_dict = {} # time step to year/month combo
    for year in years:
        for month in months:
            try:
                t100ranked = pd.read_csv(profile_file % (year, month))
            except OSError:
                pass
            else:
                t100ranked['YEAR'] = np.repeat(year,t100ranked.shape[0])
                profile_list.append(t100ranked)
                time_step_dict[time_step_count] = (year,month)
                time_step_count += 1
    #merge all profiles, assign adjaceny indices by alphabetical index on airports
    t100ranked = pd.concat(profile_list)
    t100ranked['ADJACENCY_IND'] = t100ranked.apply(lambda row: sorted([airports.index(a) for a in row['BI_MARKET'].split('_')]),axis=1)
    t100ranked['ADJ_1'] = t100ranked.apply(lambda row: row['ADJACENCY_IND'][0],axis=1)
    t100ranked['ADJ_2'] = t100ranked.apply(lambda row: row['ADJACENCY_IND'][1],axis=1)
    t100ranked['COST_PER_HOUR'] = t100ranked['FLIGHT_COST']/t100ranked['FLIGHT_TIME']
    #carrier average costs per hour, add to main data table
    ###OPTIONALLY, WEIGHT  COST AVERAGES BY FREQUENCY. CONSIDER LOOKAHEAD AS WELL FOR CRF INFERENCE
    carrier_cost_per_hour_avgs = t100ranked[['UNIQUE_CARRIER','COST_PER_HOUR','YEAR','MONTH']].groupby(['UNIQUE_CARRIER','YEAR','MONTH']).aggregate(np.mean).reset_index()
    carrier_cost_per_hour_avgs['COST_PER_HOUR_CARRIER_AVG'] = carrier_cost_per_hour_avgs['COST_PER_HOUR']
    carrier_cost_per_hour_avgs = carrier_cost_per_hour_avgs.drop('COST_PER_HOUR',axis = 1)
    t100ranked = pd.merge(t100ranked,carrier_cost_per_hour_avgs,on=['UNIQUE_CARRIER','YEAR','MONTH'])
    #get unique carrier
    carriers = t100ranked['UNIQUE_CARRIER'].unique().tolist()
    #group for carrier by carrier adj matrices
    t100ranked_grouped = t100ranked.groupby(['YEAR','MONTH','UNIQUE_CARRIER'])
    #compress markets across carriers
    t100_by_market = t100ranked[['YEAR','MONTH','MARKET_TOT','ADJ_1','ADJ_2']].groupby(['YEAR','MONTH','ADJ_1','ADJ_2']).aggregate({'MARKET_TOT':lambda x: x.iloc[0]}).reset_index()
    t100_by_market_grouped = t100_by_market.groupby(['YEAR','MONTH'])   
    
    
    print("creating demand  and avg cost tensors...")
    ###IF COSTS ARE EVER FIXED, USE PLACE TO COMBINE OBSERVED WITH HYPOTHETICAL (IN ALL PLAYED MARKETS) COSTS
    t100_by_market = t100ranked[['YEAR','MONTH','MARKET_TOT','ADJ_1','ADJ_2']].groupby(['YEAR','MONTH','ADJ_1','ADJ_2']).aggregate({'MARKET_TOT':lambda x: x.iloc[0]}).reset_index()
    t100_by_market_grouped = t100_by_market.groupby(['YEAR','MONTH'])    
    demand_tensor = np.zeros([len(airports),len(airports),time_step_count])    
    for i in range(0,time_step_count):
            year = time_step_dict[i][0]
            month = time_step_dict[i][1]
            #get relevant data table
            data_time_t = t100_by_market_grouped.get_group((year,month))
            #convert to networkx undirected graph
            G=nx.from_pandas_dataframe(data_time_t, 'ADJ_1', 'ADJ_2', edge_attr=['MARKET_TOT'])
            nx.write_edgelist(G,'netx_objs/carrier_net_demand_%s_%s.edgelist' % (year,month), data=['MARKET_TOT'])
            #convert to numpy adjacency matrix weighted by frequency
            D = nx.to_numpy_matrix(G, nodelist = range(0,len(airports)),weight='MARKET_TOT')            
            #add matrices to tensor at appropriate time step
            demand_tensor[:,:,i] = D            
           
            
            print("%s %s  demand adj mats done" % (year, month))
    np.save('demand_tensors/ts_demand',demand_tensor )
    #create carrier frequency tensor
    for carrier in carriers:
        #initialize adjaceny tensor
        adjacency_tensor = np.zeros([len(airports),len(airports),time_step_count])
        cost_tensor = np.zeros([len(airports),len(airports),time_step_count])
        
        for i in range(0,time_step_count):
            year = time_step_dict[i][0]
            month = time_step_dict[i][1]
            #get relevant data table
            try:
                carrier_time_t = t100ranked_grouped.get_group((year,month,carrier))
            except: 
                pass
            else:
                #convert to networkx undirected graph
                G=nx.from_pandas_dataframe(carrier_time_t, 'ADJ_1', 'ADJ_2', edge_attr=['DAILY_FREQ', 'FLIGHT_COST'])
                nx.write_edgelist(G,'netx_objs/carrier_net_%s_%s_%s.edgelist' % (year,month,carrier), data=['DAILY_FREQ', 'FLIGHT_COST'])
                #convert to numpy adjacency matrix weighted by frequency
                A = nx.to_numpy_matrix(G, nodelist = range(0,len(airports)),weight='DAILY_FREQ')
                #convert to numpy adjacency matrix weighted by cost
                C = nx.to_numpy_matrix(G, nodelist = range(0,len(airports)),weight='FLIGHT_COST')
                #add matrices to tensor at appropriate time step
                adjacency_tensor[:,:,i] = A
                cost_tensor[:,:,i] = C
            #next time step
            
            print("%s %s %s adj mats done" % (year, month, carrier))
                
        # save tensors
        np.save('freq_tensors/ts_freq_%s' % carrier,adjacency_tensor )
        #Cost observed. ###HAS SOME STRANGE VALUES. EVENTUALLY INCLUDE HYPOTHETICAL MARKETS AS WELL
        np.save('cost_tensors/ts_cost_%s' % carrier,cost_tensor )
    #get demand tensor (compress across t100ranked across markets)


def time_series_analysis():
    carriers_of_interest = ['WN','DL','UA','AA','US']
    airports = sorted(node_sets['top100_2014'])
    entry_dict = {}
    total_entries = 0
    entry_zero_offset = 12 # assume that if entry, airline has been out six months
    entry_mark
    for carrier in carriers_of_interest:
        adjacency_tensor = np.load('freq_tensors/ts_freq_%s.npy' % carrier )
        # is airline ever present in market?
        #create C-ordered flattened index
        flattened_index = [(i,j) for i in airports for j in airports ]
        market_presence_mat = adjacency_tensor.sum(axis=2).astype(bool)
        #don't double count markets!
        market_presence_mat[np.tril_indices(len(airports))] =False
       
        flattened_freq = adjacency_tensor.reshape(-1, adjacency_tensor.shape[-1])
        ###flattened_cost = cost_tensor.reshape(-1, cost_tensor.shape[-1])
        flattened_market_presence_index = market_presence_mat.flatten()
        freq_series_text_index = np.array(flattened_index)[flattened_market_presence_index]
        #valued time series
        FREQ_SERIES= flattened_freq[flattened_market_presence_index]
        ###COST_SERIES = flattened_cost[flattened_market_presence_index]
       
        
        #NEED MORE EFFICIENT NUMPY-ESQUE WAY OF DOING THIS   
        time_steps = FREQ_SERIES.shape[1]
        viable_markets_count = FREQ_SERIES.shape[0]
        entries = np.zeros([viable_markets_count,time_steps])
        for i in range(0,viable_markets_count):       
            for j in range(0,time_steps-1):            
                if j < entry_zero_offset:
                    pass
                    ##previous_freqs_sum = sum(FREQ_SERIES[i,0:j+1])
                else:
                    previous_freqs_sum =sum(FREQ_SERIES[i,j-(entry_zero_offset):j+1])
                    if previous_freqs_sum == 0 and FREQ_SERIES[i,j+1] > 1:
                        entries[i,j+1] = 1
        '''
        entry_ind = entries.sum(axis=1).astype(bool)
        plt.plot(np.tile(list(range(0,105)),[FREQ_SERIES.shape[0],1])[entry_ind,:].T,FREQ_SERIES[entry_ind,:].T)
        plt.xlabel('Time Step')
        plt.ylabel('Daily Frequency')
        plt.title('124 WN markets with entry, 2007-2015')
        '''
        new_entries = entries.sum().sum()
        total_entries += new_entries
        print('%s has %s entries' % (carrier, new_entries))
        entry_dict[carrier] = {'num_entries':new_entries,'mkt_index':freq_series_text_index,'freq_mat': FREQ_SERIES, 'entry_mat':entries }
    
    print(total_entries)
    
    
    '''
    plot market entry counts
    h = entries.sum(axis=0)
    plt.plot(range(0,len(h)),h)
    plt.xlabel('Time Step')
    plt.ylabel('Number of Entries')
    plt.title('WN entry counts per month, 2007-2015')
    '''
def feature_construction():
    test_time_step = 94
    carrier_of_interest = 'WN'
    training_timestep_lag = 12
    airports = sorted(node_sets['top100_2014'])
    t100ranked = pd.read_csv("full_t100ranked.csv")
    carriers = t100ranked['UNIQUE_CARRIER'].unique().tolist()
    carriers.remove(carrier_of_interest)
    freq_tensor_of_interest  = np.load('freq_tensors/ts_freq_%s.npy' % carrier_of_interest)
    other_weighted_tensor = np.zeros([freq_tensor_of_interest.shape[0],freq_tensor_of_interest.shape[1],freq_tensor_of_interest.shape[2] ])
    other_unweighted_tensor = np.zeros([freq_tensor_of_interest.shape[0],freq_tensor_of_interest.shape[1],freq_tensor_of_interest.shape[2] ])
    for other in carriers:
        ts =  np.load('freq_tensors/ts_freq_%s.npy' % other)
        other_weighted_tensor += ts
        other_unweighted_tensor += ts.astype(bool).astype(float)
    demand_ts = np.load('demand_tensors/ts_demand.npy')
    DATA_MAT = np.zeros([training_timestep_lag*freq_tensor_of_interest.shape[0]*(freq_tensor_of_interest.shape[0]-1)/2, 11  ])
    i = 0
    for timestep in range(test_time_step - training_timestep_lag,test_time_step):
        print(timestep)
        G=nx.from_numpy_matrix(freq_tensor_of_interest[:,:,timestep-1])
        deg_cent = nx.degree_centrality(G)
        for node_pair in combinations(range(0,len(airports)),2):
        
            #presence or abscence of link by airline of interest at time t
            DATA_MAT[i,0] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep].astype(bool).astype(float)
            #degree centrality of each node
            DATA_MAT[i,1] = deg_cent[node_pair[0]] + deg_cent[node_pair[1]] 
            #indicator of presence of 0, 1 or 2 nodes with non zero degree
            if deg_cent[node_pair[0]] == 0  and  deg_cent[node_pair[1]] ==0:
                DATA_MAT[i,2]  = 0.
            elif deg_cent[node_pair[0]] > 0  and  deg_cent[node_pair[1]]  > 0:
                DATA_MAT[i,2]  = 2.
            else:
                DATA_MAT[i,2]  = 1.
            #demand previous time step
            DATA_MAT[i,3] = demand_ts[node_pair[0],node_pair[1],timestep -1]
            #demand current time step
            DATA_MAT[i,4] = demand_ts[node_pair[0],node_pair[1],timestep ]
            #number of competitors on edge
            DATA_MAT[i,5] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep -1 ]
            #competitor frequency  on edge
            DATA_MAT[i,6] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep - 1 ]
            #number of competitors on edge 2 time steps ago 
            DATA_MAT[i,7] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep -2 ]
            #competitor frequency  on edge
            DATA_MAT[i,8] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep - 2 ]
             # weight of link by carrer in previous time step
            DATA_MAT[i,9] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep-1]
            DATA_MAT[i,10] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep-1].astype(bool).astype(float)
            i+=1
    np.save('DATA_MAT.npy',DATA_MAT)
    #Create testing data
    TEST_MAT = np.zeros([freq_tensor_of_interest.shape[0]*(freq_tensor_of_interest.shape[0]-1)/2, 11 ])
    i = 0
    timestep=test_time_step    
    G=nx.from_numpy_matrix(freq_tensor_of_interest[:,:,timestep-1])
    deg_cent = nx.degree_centrality(G)
    for node_pair in combinations(range(0,len(airports)),2):
    
        #presence or abscence of link by airline of interest at time t
        TEST_MAT[i,0] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep].astype(bool).astype(float)
        #degree centrality of each node
        TEST_MAT[i,1] = deg_cent[node_pair[0]] + deg_cent[node_pair[1]] 
        #indicator of presence of 0, 1 or 2 nodes with non zero degree
        if deg_cent[node_pair[0]] == 0  and  deg_cent[node_pair[1]] ==0:
            TEST_MAT[i,2]  = 0.
        elif deg_cent[node_pair[0]] > 0  and  deg_cent[node_pair[1]]  > 0:
            TEST_MAT[i,2]  = 2.
        else:
            TEST_MAT[i,2]  = 1.
        #demand previous time step
        TEST_MAT[i,3] = demand_ts[node_pair[0],node_pair[1],timestep -1]
        #demand current time step
        TEST_MAT[i,4] = demand_ts[node_pair[0],node_pair[1],timestep ]
        #number of competitors on edge
        TEST_MAT[i,5] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep -1 ]
        #competitor frequency  on edge
        TEST_MAT[i,6] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep - 1 ]
        #number of competitors on edge 2 time steps ago 
        TEST_MAT[i,7] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep -2 ]
        #competitor frequency  on edge
        TEST_MAT[i,8] = other_unweighted_tensor[node_pair[0],node_pair[1],timestep - 2 ]
        # weight of link by carrer in previous time step
        TEST_MAT[i,9] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep-1]
        TEST_MAT[i,10] = freq_tensor_of_interest[node_pair[0],node_pair[1],timestep-1].astype(bool).astype(float)
        i+=1
    np.save('TEST_MAT.npy',TEST_MAT)

    #logistic regrssion    
   
    Y = DATA_MAT[:,0].flatten()
    X = DATA_MAT[:,1:]
    
    
    # create a multinomial logistic regression model with l2 regularization using sci-kit learn
    logreg = linear_model.LogisticRegression(class_weight ='auto')
    #fit model with data
    logreg.fit(X,Y)
    #get model predictions
    Ytest = TEST_MAT[:,0]
    Yhat = logreg.predict_proba(TEST_MAT[:,1:])
    Yhatlab = logreg.predict(TEST_MAT[:,1:])
    confusion_matrix(Yhatlab,Ytest)
    '''
    array([[4415,   23],
       [  11,  501]])
       '''
       
       #ADJUST THRESHOLD? look at coefs as well!
    entries = entry_dict[carrier_of_interest]['entry_mat']
    mkt_index = entry_dict[carrier_of_interest]['mkt_index']
    #entry markets?
    entry_mkts = mkt_index[entries[:,test_time_step].astype(bool)]
    '''
    [['ATL', 'BDL'],
       ['ATL', 'DAL'],
       ['BNA', 'DAL'],
       ['DAL', 'FLL'],
       ['DAL', 'LGA'],
       ['DAL', 'PHX'],
       ['DAL', 'SAN'],
       ['DAL', 'TPA'],
       ['DCA', 'IND']], '''
    full_markets =  list(combinations(range(0,len(airports)),2))
    entry_markets_nums = [(airports.index(mkt[0]),airports.index(mkt[1])) for mkt in entry_mkts]
    Yhat_entry_index  = [full_markets.index(entry) for entry in entry_markets_nums]
    entry_probs = Yhat[Yhat_entry_index]
    '''
    array([[ 0.8292276 ,  0.1707724 ],
       [ 0.81331235,  0.18668765],
       [ 0.73079617,  0.26920383],
       [ 0.76508113,  0.23491887],
       [ 0.16334948,  0.83665052],
       [ 0.20897478,  0.79102522],
       [ 0.73383294,  0.26616706],
       [ 0.78017388,  0.21982612],
       [ 0.99044415,  0.00955585]])'''
      # where do non predicted fall among other non predicted?
    just_predicted_0s = np.sort(Yhat[np.invert(Yhatlab.astype(bool))][:,0] , axis =0 )
    ranks_in_unpredicted = [np.where(just_predicted_0s==entry_prob[0]) for entry_prob in entry_probs]
    '''
    [(array([21], dtype=int64),),
 (array([19], dtype=int64),),
 (array([5], dtype=int64),),
 (array([9], dtype=int64),),
 (array([], dtype=int64),),
 (array([], dtype=int64),),
 (array([6], dtype=int64),),
 (array([11], dtype=int64),),
 (array([4108], dtype=int64),)]
 
 
 len(just_predicted_0s)
Out[411]: 4438
 '''
 
 
 
 
def run():
    for year in range(2007,2016):
        t = nonstop_market_profile_monthly(output_file = "monthly_market_profile_%sm%s.csv",year = year, months=range(1,13), \
    t100_fn="bts_data/T100_%s.csv" % year,p52_fn="bts_data/SCHEDULE_P52_%s.csv" % year, t100_avgd_fn="processed_data/t100_avgd_m%s.csv", merge_HP=True, \
    t100_summed_fn = 'processed_data/t100_summed_m%s.csv', t100_craft_avg_fn='processed_data/t100_craft_avg_m%s.csv',\
    ignore_mkts = [], craft_freq_cuttoff = .01,max_competitors=100,\
    freq_cuttoff = .5, ms_cuttoff=.05, fs_cuttoff = .05, only_big_carriers=False, airports = node_sets['top100_2014'])
        