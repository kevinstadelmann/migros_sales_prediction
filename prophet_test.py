# %%
#******************************************#
#   Import and load files
#******************************************#
import prophet as pr
import pandas as pd

# load data
df_raw = pd.read_csv("data_raw/data_raw_merged.csv")
df_proph = df_raw[df_raw["article_name"] == "AHA JOGHURT LAKTOSEF CLAS"].copy()

# only keep necessary columnes
df_proph = df_proph[['date','sales']]

# Prophet needs the following two columnes to work
df_proph.columns = ['ds', 'y']
df_proph['ds']= pd.to_datetime(df_proph['ds'])


# %%
#******************************************#
#   Create Prophet Model
#******************************************#
# Test Prophet

# define the model
model = pr.Prophet()
# fit the model
model.fit(df_proph)

# %%
#******************************************#
#   Values for prediction
#******************************************#

# create values for prediction
# in-sample forecast
future = ("2022-06-20","2022-06-27","2022-07-04","2022-07-11","2022-07-18")
future = pd.DataFrame(future)
future.columns = ['ds']
future['ds']= pd.to_datetime(future['ds'])



# %%
#******************************************#
#   Execute Prediction
#******************************************#

# summarize the forecast
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())


# %%

model.plot(forecast)
# %%
model.plot_components(forecast)

# %%

from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(model, forecast)
# %%
