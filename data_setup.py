# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:34:48 2016

@author: d29905p
"""



'''
premiinary data setup and exploration

files needed:

'''

import os 
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import networkx as nx

data_dir = "C:/Users/d29905P/Documents/airline_competition_paper/code/network_games/bts_data/"

output_dir = "C:/Users/d29905P/Documents/airline_comp_networks/data/adjacency_mats/"


node_sets = {'ope35_sansHLN' : ['ATL', 'BOS', 'BWI', 'CLE', 'CLT', 'CVG', 'DCA', 'DEN', 'DFW', 'DTW', 'EWR', 'FLL', 'IAD', 'IAH', 'JFK', 'LAS', 'LAX', 'LGA', 'MCO', 'MDW', 'MEM', 'MIA', 'MSP', 'ORD', 'PDX', 'PHL', 'PHX', 'PIT', 'SAN', 'SEA', 'SFO', 'SLC', 'STL', 'TPA'],
'western': ['SEA','PDX','SFO','SAN','LAX','LAS','PHX','OAK','ONT','SMF','SJC'],
'top100_2014':['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'IAH', 'SFO', 'PHX', 'LAS', 'SEA', 'MSP', 'BOS', 'SLC', 'EWR', 'MCO', 'DTW', 'LGA', 'CLT', 'JFK', 'MDW', 'BWI', 'SAN', 'MIA', 'PHL', 'DCA', 'TPA', 'HOU', 'FLL', 'IAD', 'PDX', 'STL', 'BNA', 'HNL', 'MCI','OAK', 'DAL', 'AUS', 'SJC', 'SMF', 'RDU', 'SNA', 'MSY', 'MKE', 'SAT', 'CLE', 'IND', 'PIT', 'SJU', 'ABQ', 'CMH', 'OGG', 'OKC', 'BDL', 'ANC', 'BUR', 'JAX', 'CVG', 'ONT', 'OMA', 'TUL', 'RIC', 'ELP', 'RSW', 'BUF', 'RNO', 'PBI', 'CHS', 'LIT', 'TUS', 'MEM','SDF', 'BHM', 'LGB', 'PVD', 'KOA', 'BOI', 'GRR', 'LIH', 'ORF', 'FAT', 'XNA', 'DAY', 'MAF', 'GEG', 'MSN', 'DSM', 'COS', 'GSO', 'TYS', 'ALB', 'SAV', 'PNS', 'BTR', 'ICT', 'ROC', 'JAN', 'MHT', 'AMA','FSD', 'HPN']}


#add top 100? or look at graph, consider 2007, and summer 2014 --set diff

def create_airport_traffic_rank():
    #parameters
    file_recent = data_dir + "AOTPjuly_2014.csv"
    
    aotp_recent =  pd.read_csv(file_recent)
    
    departure_rank = aotp_recent.ORIGIN.value_counts()
    plt.plot(range(0,len(departure_rank.values)),departure_rank.values)
    #in 2014, top 100 airports-> 90 percent of flights
    
    
def create_tensor_daily_freq():
    #parameters
    years = [2007,2008]
    months = range(1,13)
    smoothing = 9
    airports = 'ope35_sansHLN'
    
    if airports != 'all': 
        node_names = sorted(node_sets[airports])
    #OTHERWISE LOAD FROM YEARLY DATA
    
    for year in years:
        for month in months: 
            pass





    year=2007
    marketset = sorted(node_sets[airports])
    include_CX_DIV = True
    #rolling average lag in days
    smoothing = 9
    aotp = pd.read_csv(data_dir + "aotp%s.csv" % year, usecols = ['YEAR','QUARTER','MONTH','DAY_OF_MONTH','FLIGHT_DATE','ORIGIN','DESTINATION','UNIQUE_CARRIER','CANCELLED','DIVERTED','NUMBER_FLIGHTS','TAIL_NUMBER' ])    
    #select relevant markets
    aotp = aotp[aotp['ORIGIN'].isin(marketset) & aotp['DESTINATION'].isin(marketset)]
    aotp_perday = aotp[['YEAR','QUARTER','MONTH','DAY_OF_MONTH','FLIGHT_DATE','ORIGIN','DESTINATION','UNIQUE_CARRIER','NUMBER_FLIGHTS']].groupby(['YEAR','QUARTER','MONTH','DAY_OF_MONTH','FLIGHT_DATE','ORIGIN','DESTINATION','UNIQUE_CARRIER']).aggregate(np.sum).reset_index()
    #create date index for this year    
    date_index = create_date_index_dict(year)
    #create reverse map
    reverse_date_index = {value:key for key, value in date_index.items()}
    #add date index for dataframe
    aotp_perday['DATE_INDEX'] =aotp_perday.apply(lambda row: date_index[row['FLIGHT_DATE']],1)
    #function to create zero flight padded full year index from numpy arrays of flights and dates
    num_days = len(date_index)
    def yearly_flight_count(DATE_INDEX,NUM_FLIGHTS,num_days):
        num_flights_padded = np.zeros(num_days)
        num_flights_padded[DATE_INDEX]=NUM_FLIGHTS
        return num_flights_padded
    #reshape matrix by collapasing days into vector
    market_dynamics_rows = []
    aotp_perday_gb = aotp_perday[['ORIGIN','DESTINATION','UNIQUE_CARRIER','NUMBER_FLIGHTS','DATE_INDEX']].groupby(['ORIGIN','DESTINATION','UNIQUE_CARRIER'])
    for group in aotp_perday_gb.groups:
        DATE_INDEX = aotp_perday_gb.get_group(group)['DATE_INDEX']
        NUM_FLIGHTS = aotp_perday_gb.get_group(group)['NUMBER_FLIGHTS']
        flights_vec = yearly_flight_count(DATE_INDEX,NUM_FLIGHTS,num_days)
        market_dynamics_rows.append({'UNIQUE_CARRIER': group[2],'ORIGIN':group[0],'DESTINATION':group[1],'FLIGHTS_VEC':flights_vec})
    market_dynamics_df = pd.DataFrame(market_dynamics_rows)
    
    
    #add rolling avg of flights
    rolling = []
    for vec in market_dynamics_df['FLIGHTS_VEC'].tolist():        
        rolling.append(pd.rolling_mean(vec,smoothing) )
        
    market_dynamics_df['ROLLING'] =rolling
    market_dynamics_df['ROLLING_MAX']=market_dynamics_df.apply(lambda row: np.nanmax(row['ROLLING']),1)
    
    
# create zero tensor for US
    freq_cuttoff = 2
    US_flights = market_dynamics_df[market_dynamics_df.UNIQUE_CARRIER=='US'].reset_index()
    US_tensor  =  np.zeros([len(marketset), len(marketset),365])
    for i in range(0,US_flights.shape[0]):
        origin_ind = marketset.index(market_dynamics_df.loc[i,'ORIGIN'])
        dest_ind = marketset.index(market_dynamics_df.loc[i,'DESTINATION'])        
        US_tensor[origin_ind,dest_ind,:] = US_flights.loc[i,'ROLLING']
    US_tensor = np.nan_to_num(US_tensor)
    US_tensor[US_tensor<freq_cuttoff]=0
    #symmetrize (EVENTUALLY CHECK IF APPROPRIATE?)
    for i in range(0,365):
        US_tensor[:,:,i] = (US_tensor[:,:,i] + US_tensor[:,:,i].T)/2
    
    G1=nx.from_numpy_matrix(US_tensor[:,:,20])
    G2=nx.from_numpy_matrix(US_tensor[:,:,360])
    entries = nx.difference(G2,G1)
    exits=nx.difference(G1,G2)
    
    labels  = {i:lab for i, lab in enumerate(marketset)}    
    
    location_data = pd.read_csv(data_dir + "airports_locations.dat", header=None).set_index(4)
    
    pos = {i:[location_data.loc[lab][7],location_data.loc[lab][6]] for i, lab in enumerate(marketset)}    
    plt.figure()
    plt.ylim([20,80])
    plt.xlim([-130,-70])
    plt.title('US Airways network, January 2007' )
    plt.axis('equal')
    nx.draw_networkx(G1, pos = pos,labels = labels, font_size=8, node_size=500, edge_color='b')
    #nx.draw_networkx_labels(G1, pos = pos,labels = labels)
    plt.figure()
    plt.ylim([20,80])
    plt.xlim([-130,-70])
    plt.title('US Airways network, December 2007' )
    plt.axis('equal')
    nx.draw_networkx(nx.difference(G1,exits), pos = pos,labels = labels, font_size=8, node_size=500, edge_color='b')
    nx.draw_networkx_edges(exits, pos = pos, font_size=8, node_size=500, edge_color='b',style='dashed')
    nx.draw_networkx_edges(entries, pos = pos, font_size=8, node_size=500, edge_color='g')
    #nx.draw_networkx_labels(G2, pos = pos, labels = labels)
    
    
    market_dynamics_groups = market_dynamics_df.groupby(['ORIGIN','DESTINATION'])
    #plots for unidrectional markets (check discripency, eliminate max <1?)
    #create date index and month labels
    xvec =np.array(range(0,num_days))
    month_labs=[]
    lab_locs = []
    month_prev = 'NULL'
    loc= 0
    for date_ind in xvec.tolist():
        month = reverse_date_index[date_ind].split('-')[1]
        if month!=month_prev:
            month_labs.append(month)
            lab_locs.append(loc)
        #else:
           # month_labs.append('')
        month_prev=month      
        loc += 1

'''
not a priority, probably better to use composition estimates for each carrier/segment
'''
def create_tensor_daily_freq_craft_disagg():
    pass

def create_tensor_monthly_freq_aotp():
    pass 

def create_tensor_monthly_freq_t100():
    pass 

def create_tensor_quarterly_cost_craft_disagg():
    pass 

'''
weight aggregation for different craft by quarterly usuage
'''
def create_tensor_quarterly_cost_weighted_agg():
    pass 

def create_tensor_quarterly_route_demand():
    pass 

'''
almost certainly useless to disaggregate by craft, passengers don't care that much
'''
def create_tensor_monthly_seg_demand():
    pass 



#make scales dependent on max frequency!
airlines2014 = ['AA', 'AS', 'CP','DL','MQ',  'OO',  'QX',  'UA',  'US',  'VX', 'WN' ,'YV']
airlines2007 = ['AA','AS','MQ','OO','QX','UA','US','WN','HP']
def t100_monthly_viz( outdir = "C:/Users/d29905p/documents/airline_competition_paper/code/network_games/", merge_HP = True, basefn = '12month_ms-freq_', freq_cuttoff = 1, ms_cuttoff=.1, fs_cuttoff=.1, airports = ['SEA','PDX','SFO','SAN','LAX','LAS','PHX','OAK','ONT','SMF','SJC'],airlines = airlines2014,years = [2014], months=list(range(1,13))):
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    def create_market(row):
        market = [row['ORIGIN'], row['DEST']]
        market.sort()
        return "_".join(market)
    for year in years:
        t100_yr = pd.read_csv('bts_data/T100_%s.csv' % str(year))
        if airlines:
            t100_yr_network = t100_yr[t100_yr['ORIGIN'].isin(airports) & t100_yr['DEST'].isin(airports) & t100_yr['UNIQUE_CARRIER'].isin(airlines) ]
        else:
            t100_yr_network = t100_yr[t100_yr['ORIGIN'].isin(airports) & t100_yr['DEST'].isin(airports) ]
            
        t100_yr_network['BI_MARKET'] = t100_yr_network.apply(create_market,1)
        if merge_HP:
            t100_yr_network['UNIQUE_CARRIER']=t100_yr_network['UNIQUE_CARRIER'].replace('HP','US')
        #sum between craft type
        t100_yr_network_merge_mkt = t100_yr_network[['UNIQUE_CARRIER','BI_MARKET','MONTH','DEPARTURES_SCHEDULED','PASSENGERS','ORIGIN','DEST']].groupby(('UNIQUE_CARRIER','BI_MARKET','ORIGIN','DEST','MONTH')).aggregate(np.sum).reset_index()
        #sum between craft type
        t100_yr_network_merge_craft = t100_yr_network_merge_mkt[['UNIQUE_CARRIER','BI_MARKET','MONTH','DEPARTURES_SCHEDULED','PASSENGERS']].groupby(('UNIQUE_CARRIER','BI_MARKET','MONTH')).aggregate(np.mean).reset_index()        
        t100_yr_network_merge_craft['DAILY_FREQ'] = t100_yr_network_merge_craft.apply(lambda row: row['DEPARTURES_SCHEDULED']/float(days_in_month[int(row['MONTH'])-1]),1)
        t100_yr_network_merge_craft_freq_filt = t100_yr_network_merge_craft[t100_yr_network_merge_craft['DAILY_FREQ']>=freq_cuttoff]
        #t100_yr_network_merge_craft_freq_filt
        t100_grouped = t100_yr_network_merge_craft_freq_filt.groupby(('BI_MARKET','MONTH'))
        grouplist = []
        for market in list(set(t100_yr_network_merge_craft_freq_filt['BI_MARKET'].tolist())):
            for month in range(1,13):
                try:
                    market_group = t100_grouped.get_group((market,month))
                    new_group = market_rank(market_group, ms_cuttoff=ms_cuttoff, fs_cuttoff=fs_cuttoff)
                    grouplist.append(new_group)
                except KeyError:
                    pass
        t100ranked = pd.concat(grouplist,axis=0)
    '''
    add concatenation of multiple years later
    '''
   
    #plot frequencies of major carriers in different markets over course of year as well as number of competitors (especially relelvant when we do a market share cuttoff)
    t100markets = t100ranked.groupby('BI_MARKET')    
    markets = list(set(t100ranked['BI_MARKET'].tolist()))  # WHY IS THIS DIFFERENT markets = list(set(t100_yr_network_merge_craft_freq_filt['BI_MARKET'].tolist())) 
    for market in markets:
        market_df = t100markets.get_group(market)    
        plt.subplot(2,1,1)
        plt.ylabel('Flights per Day')
        for carrier in list(set(market_df['UNIQUE_CARRIER'].tolist())):
            carrier_gb = market_df[market_df['UNIQUE_CARRIER']==carrier].set_index('MONTH')
            carrier_vector = []
            for i in range(1,13):
                try:
                    carrier_vector.append(float(carrier_gb.loc[i]['DAILY_FREQ']))
                except KeyError:
                    carrier_vector.append(0.0)
            plt.plot(list(range(1,13)),carrier_vector, label=carrier)
        plt.legend(shadow=True, loc=3,fancybox=True)   
        plt.title('Frequency Competition in %s Market, %s' % (market, year))
        #plt.show()
        #ADD MARKET COMPETITOR MARKER?? SEARCH FOR ENTRIES
        plt.axis([1, 12, 0, 22])
        plt.subplot(2,1,2)
        plt.ylabel('Mkt Share')
        plt.xlabel('time (months)')
        plt.axis([1, 12, 0, 1.1])
        for carrier in list(set(market_df['UNIQUE_CARRIER'].tolist())):
            carrier_gb = market_df[market_df['UNIQUE_CARRIER']==carrier].set_index('MONTH')
            carrier_vector = []
            for i in range(1,13):
                try:
                    carrier_vector.append(float(carrier_gb.loc[i]['MS_TOT']))
                except KeyError:
                    carrier_vector.append(0.0)
            plt.plot(list(range(1,13)),carrier_vector, label=carrier)
        #plt.legend(shadow=True, loc=2,fancybox=True)       
        
        plt.savefig(outdir + 't100_%s_pics/%s_%s.jpg' % (year,basefn, market))
        plt.clf()
        
        
        
        
        
        
        
'''
date tools

'''
def create_date_index_dict(year=2007):
    
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
    #create index
    date_index = {}
    i = 0 
    #func to zero pad dates
    def zeropad(datestr):
        if len(datestr) <2:
            datestr = "0" + datestr
        return datestr
    # fill date dict
    for month in range(1,13):    
        for day  in range(1,days_month_dict[month]+1):
            date_index['-'.join([str(year),zeropad(str(month)),zeropad(str(day))])] = i
            i += 1                
    return date_index
