# %%
#******************************************#
#   Import
#******************************************#
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


# %%
#******************************************#
#   Merge all files into one file
#******************************************#
path = "C:/python_dev/customer_data_analytics/data_raw"
all_files = glob.glob(os.path.join(path,"*.csv"))
li = []

for filename in all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)


frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv(path+"/data_raw_merged.csv", index=False)

# %%
#******************************************#
#   Plot function "product sales over time"
#******************************************#

df = pd.read_csv("data_raw/data_raw_merged.csv")
df = df[df["article_name"] == "AHA JOGHURT LAKTOSEF CLAS"]

fig, ax = plt.subplots()
ax.plot(df["yearweek_start"], df["sales"])
ax.set(xlabel='Time (week)',
        ylabel='Sales',
        title='AHA JOGHURT LAKTOSEF CLAS')

plt.show()

# %%
#******************************************#
#   Plot function "product sales over time vs total sales"
#******************************************#

df = pd.read_csv("data_raw/data_raw_merged.csv")
df = df[df["article_name"] == "AHA JOGHURT LAKTOSEF CLAS"]

fig, ax = plt.subplots()
ax.plot(df["yearweek_start"], df["sales"])
ax.set(xlabel='Time (week)',
        ylabel='Sales',
        title='AHA JOGHURT LAKTOSEF CLAS')

plt.show()
# %%
