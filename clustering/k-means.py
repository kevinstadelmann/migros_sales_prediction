# %%
#******************************************#
#   Import
#******************************************#
from cProfile import label
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import tslearn.utils as ts_utils
import numpy as np
import tslearn.clustering as ts_cluster

PATH="../data_raw"

# %%

#******************************************#
# Do a boxplot to see if all the "sales" have a similiar scale
#******************************************#

full_df = pd.read_csv("../data_raw/data_raw_merged.csv")

full_df.info()
full_df['article_name'].unique()

full_df.boxplot(by ='article_name', column =['sales'], grid = False, rot=90)

# %%

#******************************************#
# Print all the poduct sales on one plot
#******************************************#

prod_list = full_df['article_name'].unique()
plt.figure(figsize=(12,6),dpi=200)

for product in prod_list:
    
    product_df = full_df[full_df['article_name'] == product]
    print(f" {product} size is {product_df['date'].count()}")
    plt.plot(product_df['date'],product_df['sales'])

plt.show()


# %%
######################################
# check how many obs for each product
######################################

prod_list = full_df['article_name'].unique()

for product in prod_list:
    product_df = full_df[full_df['article_name'] == product]
    print(f"product:{product} has {len(product_df)}")



#it seems that "M-CLAS JOG. NATURE 200G" only have sales figures from 1 July 2019  - approx 160 observations


# %%

######################################
# setup tslearn data and do elbow plot for number of clusters
######################################

prod_list = full_df['article_name'].unique()
plt.figure(figsize=(12,6),dpi=200)

ts_list = []

#removing the first product ("M-CLAS JOG. NATURE 200G") as they only have 160 observations
#  (seems to cause a weird issues with tslearn)
# TODO: investigate that further.

prod_list = np.delete(prod_list,[0])

# create an 2d array of sales values
for product in prod_list:
    product_df = full_df[full_df['article_name'] == product]
    sales = product_df['sales'].tolist()
    ts_list.append(sales)


formatted_ts_list = ts_utils.to_time_series(ts_list)
print(formatted_ts_list.shape)

# try to do an elbow graph to figure out best k value to use 
# based on the sum square distance

sum_square_dist = []

for k in range(2,18):
    km = ts_cluster.TimeSeriesKMeans(n_clusters=k,metric="dtw")
    labels = km.fit_predict(formatted_ts_list)
    #print(labels)
    sum_square_dist.append(km.inertia_)

print(sum_square_dist)
plt.plot(range(2,18),sum_square_dist)

#I think the best version of K is five or six (will go with 6 for now)

#%%
######################################
# create k means model with k=6
######################################

k = 6
km = ts_cluster.TimeSeriesKMeans(n_clusters=k,metric="dtw")
labels = km.fit_predict(formatted_ts_list)

cluster_assignment = pd.DataFrame({"Product":prod_list,
                            "label":labels
                            })
print(cluster_assignment)
print(cluster_assignment.info())


#%% 

######################################
# plot each cluster of series
######################################

def plot_cluster_label(cluster_label : int , cluster_assignment : pd.DataFrame , data_df : pd.DataFrame):
    
    plt.figure(figsize=(12,6),dpi=200)

    products_in_cluster = cluster_assignment[cluster_assignment['label'] == cluster_label]
    for current_product in products_in_cluster['Product']:
        #print the products time series
        product_df = data_df[data_df['article_name']==current_product]
        plt.plot(product_df['date'],product_df['sales'],label=current_product)
        plt.title(f"Cluster {cluster_label}")
        plt.legend()
    

plot_cluster_label(5,cluster_assignment,full_df)
plot_cluster_label(4,cluster_assignment,full_df)
plot_cluster_label(3,cluster_assignment,full_df)
plot_cluster_label(2,cluster_assignment,full_df)
plot_cluster_label(1,cluster_assignment,full_df)
plot_cluster_label(0,cluster_assignment,full_df)


######################################
# Questions
######################################
    
#Q1 Should we by scaling the sales value first? Is it more important the distance
#  between the series or how they are behaving at the time.

#Q2 Should we be including the promotion fields in the clustering or just use the sales (as I have done here)


