import pandas as pd
import matplotlib.pyplot as plt
import fbprophet
df = pd.read_html('https://raw.githubusercontent.com/diazonic/vegetables_price_forecast/main/Grlic.html')[-1]
df.drop(df.tail(1).index, inplace = True)
df['Modal Price (Rs/q)'] = df['Modal Price (Rs/q)'].astype(int)
df['date'] = df['Month Name']+'-'+df['Year']
df['date'] = pd.to_datetime(df['date'])
df_new = df[['date','Modal Price (Rs/q)']]
df_new = df_new.rename(columns={'date':'ds','Modal Price (Rs/q)':'y'})
df_new= df_new.sort_values(by='ds')
plt.plot(df_new['ds'],df_new['y'])
plt.show()
m = fbprophet.Prophet(changepoint_prior_scale=1,seasonality_prior_scale=1)
m.fit(df_new)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast,xlabel = 'Date',ylabel='Price of Potato')

plt.figure(figsize=(20,10))
plt.scatter(df_new['ds'],df_new['y'],s=5,c='k')
plt.plot(forecast['ds'],forecast['yhat'])
