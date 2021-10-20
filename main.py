import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import matplotlib.pyplot as plt
import time
import requests
import json



def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()

st.subheader('Sistem Pendukung Keputusan Pencarian Harga Rumah Berdasarkan Kriteria')

@st.cache
def load_data():
	df=pd.read_excel('data.xls')
	df=df.drop(['country'],axis=1)
	df=df[df['price']>0]
	df.rename(columns={'statezip':'zip'}, inplace=True)
	df['zip']=df['zip'].str.replace('WA','').astype(int)
	df['floors']=df['floors'].astype(int)
	df=df[df['bedrooms']>0]
	df=df[df['bathrooms']>0]
	return df

df=load_data()


st.sidebar.subheader('Property Options')
params={
'bedrooms' : st.sidebar.selectbox('Kamar',(df['bedrooms'].unique())),
'bathrooms' : st.sidebar.selectbox('Kamar Mandi',(df['bathrooms'].unique())),
'floors' : st.sidebar.selectbox('Jumlah Lantai',(df['floors'].unique())),
'sqft' : st.sidebar.slider('Luas Tanah', 800,max(df['sqft_living']),step=100),
'waterfront':1 if st.sidebar.checkbox('Tepi Laut') else 0
}



def map_df(df):
	df=df[df['bedrooms']==params['bedrooms']]
	df=df[df['bathrooms']==params['bathrooms']]
	df=df[df['floors']==params['floors']]
	df=df[df['waterfront']==params['waterfront']]
	df=df[(df['sqft_living']>0.9*params['sqft']) & (df['sqft_living']<1.1*params['sqft'])]
	df.reset_index()
	return df

# test_size=st.sidebar.slider('Silahkan Pilih Tentukan Test Size', 0.05,0.5,0.25,step=0.05)
test_size = 0.25

@st.cache
def get_models():
	y=df['price']
	X=df[['bedrooms','bathrooms','floors','sqft_living','waterfront']]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
	models = [
			#  DummyRegressor(strategy='mean'),
			#    RandomForestRegressor(n_estimators=170,max_depth=25),
			   DecisionTreeRegressor(max_depth=25)
			#    GradientBoostingRegressor(learning_rate=0.01,n_estimators=200,max_depth=5), 
			#  LinearRegression(n_jobs=10, normalize=True)
			   ]
	df_models = pd.DataFrame()
	temp = {}
	print(X_test)
	for model in models:
		print(model)
		m = str(model)
		temp['Model'] = m[:m.index('(')]
		model.fit(X_train, y_train)
		temp['RMSE_Price'] = sqrt(mse(y_test, model.predict(X_test)))
		temp['Pred Value']=model.predict(pd.DataFrame(params,  index=[0]))[0]
		print('RMSE score',temp['RMSE_Price'])
		df_models = df_models.append([temp])
	df_models.set_index('Model', inplace=True)
	pred_value=df_models['Pred Value'].iloc[[df_models['RMSE_Price'].argmin()]].values.astype(float)
	return pred_value, df_models

def run_data():
	#run_status()
	df_models=get_models()[0][0]
	st.write('Prediksi Harga Rumah Rata-Rata **${:.2f}**'.format(df_models))
	df1=map_df(df)
	df1
	# st.map(df1)
	# df1[['price','bedrooms','bathrooms','floors','sqft_living','waterfront']]

def show_ML():
	df_models=get_models()[1]
	df_models
	st.write('**This diagram shows root mean sq error for all models**')
	st.bar_chart(df_models['RMSE_Price'])

btn = st.sidebar.button("Predict")
if btn:
	run_data()
else:
	pass

if st.sidebar.button('Show JSON'):
	df_models=get_models()[0][0]
	st.json(map_df(df[['price','bedrooms','bathrooms','floors','sqft_living','waterfront']]).to_json())

if st.sidebar.button('Close JSON'):
	run_data()

st.sidebar.subheader('Informasi Tambahan')
if st.sidebar.checkbox('Show ML Models'):
	run_data()
	df_models=get_models()[1]
	df_models
	# st.write('**root mean sq error for all models**')
	# st.bar_chart(df_models['RMSE_Price'])

# if st.sidebar.checkbox('Tunjukan Data Asli'):
# 	df
