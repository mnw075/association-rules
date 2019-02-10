# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 21:05:22 2019

@author: mwon579
"""
import pandas as pd
import numpy as np

import random

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

'''
Read Data
'''
filename = 'deptinfo.csv'
deptinfo = pd.read_csv(filename, header = None, usecols = range(2))
deptinfo.columns = ['dept', 'deptdesc']

filename = 'strinfo.csv'
strinfo = pd.read_csv(filename, header = None, usecols = range(4))
strinfo.columns = ['store', 'city', 'state', 'zip']

#  trnsact cols = ['sku','store','register','trannum','seq','saledate','stype','quantity','amt','amt', 'seq']
CO_stores = list(strinfo.loc[strinfo['state'] == 'CO'].store)
filename = 'trnsact.csv'
df_chunk = pd.read_csv(filename, chunksize=1000000, header = None, usecols = [0,1,2,3,5,6,7,9])
chunk_list = []  # append each chunk df here 
# Each chunk is in df format
for chunk in df_chunk:  
    # perform data filtering 
    # chunk_filter = chunk_preprocessing(chunk)
    
    # Once the data filtering is done, append the chunk to list
    chunk.columns = ['sku','store','register','trannum','saledate','stype','quantity','amt']
    chunk_list.append(chunk.loc[chunk['store'].isin(CO_stores)])
    
# concat the list into dataframe 
trnsact = pd.concat(chunk_list)



trnsact.saledate = trnsact.saledate.astype('category')
trnsact.saledate = trnsact.saledate.cat.codes



trnsact['trns'] = trnsact['store'].astype(str)+trnsact['register'].astype(str)+trnsact['trannum'].astype(str)+ trnsact['saledate'].astype(str)
trnsact.trns = trnsact.trns.astype('category')
trnsact.trns = trnsact.trns.cat.codes
trnsact.drop(columns = ['register','trannum','saledate'], inplace = True)

trnsact.to_csv('CO.csv', index = False)

trnsact = pd.read_csv('CO.csv')

#filename = 'skuinfo.csv'
#skuinfo = pd.read_csv(filename, header = None)
#skuinfo.columns = ['dept','classid','upc','style','color','size','packsize','vendor','brand']

filename = 'skstinfo.csv'
skstinfo = pd.read_csv(filename, header = None, usecols = range(3))
skstinfo.columns = ['sku', 'store','cost']

skstinfo = skstinfo.loc[skstinfo.sku.isin(trnsact.sku.unique()) & skstinfo.store.isin(trnsact.store.unique())]

# inner join the sku data table with the transactions data table so that we can compute profit
merged = pd.merge(skstinfo, trnsact, how='inner', on=['sku', 'store'])
merged['profit'] = merged.amt - merged.cost


merged.to_csv('merged.csv', index = False)


merged = pd.read_csv('merged.csv')

merged.stype.value_counts()

sku_groupby = merged.groupby('sku')

num_unique = merged.trns.nunique()


margins = sku_groupby.agg('mean').sort_values('profit').profit.reset_index()
profitable_skus = list(margins[margins.profit >= 1].sku)

sku_counts = merged.sku.value_counts().reset_index()
sku_counts.columns = ['sku','count']
skus_of_interest = list(sku_counts[(sku_counts.sku.isin(profitable_skus)) & (sku_counts['count'] > num_unique*.0004)].sku)
trns_of_interest = merged[merged.sku.isin(skus_of_interest)]
trns_of_interest.drop(columns=['store', 'cost', 'amt','profit'], inplace=True)

basket = (trns_of_interest.groupby(['trns', 'sku'])['quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('trns'))



def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

frequent_itemsets = apriori(basket_sets, min_support=0.0001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

rules = rules.sort_values('lift', ascending=False)
rules.reset_index(inplace = True)

sku_list = []
for i in range(len(rules)):
    
    ant = list(i for i in rules.antecedents[i])
    con = list(i for i in rules.consequents[i])
    sku_list += ant + con
    rules.at[i,'antecedents'] = ant
    rules.at[i,'consequents'] = con

rules.to_csv('rules.csv', index = False)

num_uniq_skus = len(set(sku_list[0:670]))

candidates = set(sku_list[0:670])

best_rules = rules

for i in range(len(rules)):
    if any(e for e in rules.antecedents[i] if e in candidates) & any(e for e in rules.consequents[i] if e in candidates):
        continue
    else:
        best_rules.drop([i], inplace = True)



best_rules.to_csv('best_rules.csv', index = False)

