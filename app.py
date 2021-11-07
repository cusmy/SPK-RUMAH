import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import matplotlib.pyplot as plt
import time
import requests
import json
from managedb import *
import hashlib


def generate_hash(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hash(password,hashed_text):
	if generate_hash(password) == hashed_text :
		return hashed_text
	return False


def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()

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

@st.cache
def get_locations(zip):
	for i in zip:
		url='https://data.opendatasoft.com/api/records/1.0/search/?dataset=geonames-postal-code%40public&q=&rows=10&facet=postal_code&refine.country_code=US&refine.postal_code='+str(i).format(zip)
		data=requests.get(url).json()
		lat=data['records'][0]['fields']['latitude']
		lng=data['records'][0]['fields']['longitude']
	return lat, lng



def map_df(df):
	df=df[df['bedrooms']==kamar]
	df=df[df['bathrooms']==kmandi]
	df=df[df['floors']==jlantai]
	df=df[df['waterfront']==tlaut]
	df=df[(df['sqft_living']>0.9*ltanah) & (df['sqft_living']<1.1*ltanah)]
	df.reset_index()
	return df

test_size = 0.25

def create_param(df):
	kamar = st.selectbox('Kamar.',(df['bedrooms'].unique()))
	kmandi = st.selectbox('Kamar Mandi.',(df['bathrooms'].unique()))
	jlantai = st.selectbox('Jumlah Lantai.',(df['floors'].unique()))
	ltanah = st.slider('Luas Tanah.', 800,max(df['sqft_living']),step=100)
	tlaut = 1 if st.checkbox('Tepi Laut.') else 0
	return kamar,kmandi,jlantai,ltanah,tlaut
@st.cache
def get_models():
	y=df['price']
	X=df[['bedrooms','bathrooms','floors','sqft_living','waterfront']]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
	models = DecisionTreeRegressor(max_depth=25)
	df_models = pd.DataFrame()
	temp = {}
	print(models)
	m = str(models)
	temp['Model'] = m[:m.index('(')]
	models.fit(X_train, y_train)
	temp['RMSE_Price'] = sqrt(mse(y_test, models.predict(X_test)))
	temp['Pred Value']=models.predict(pd.DataFrame(df,  index=[0]))[0]
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

def main():
	submenu = ["Home","SPK","Add User","Lihat User"]
	username = st.sidebar.text_input("Masukan Username")
	password = st.sidebar.text_input("Masukan Password",type='password')
	if st.sidebar.checkbox("Login"):
		create_usertable()
		hash_pasword = generate_hash(password)
		result = login_user(username,verify_hash(password,hash_pasword))
		if result:
			st.success("Welcome {}".format(username))
				
			if username == "admin":
				pilih = st.sidebar.selectbox("Menu",submenu)
				if pilih == "Home":
					st.subheader('Kelompok 2')
					st.subheader('Pencarian Harga Rumah Berdasarkan Kriteria')
					st.write("Ini adalah web untuk melakukan analisa terhadap harga rumah berdasarkan kriteria yang anda inginkan.")
					st.write("berikut adalah dataset yang sudah kami kumpulkan untuk melakukan analisa rumah menggunakan metode Decision tree")
					df=load_data()
					df
				elif pilih == "SPK":
					df=load_data()
					st.subheader('Property Options')

					params={
					'bedrooms' : st.selectbox('Kamar.',(df['bedrooms'].unique())),
					'bathrooms' : st.selectbox('Kamar Mandi.',(df['bathrooms'].unique())),
					'floors' : st.selectbox('Jumlah Lantai.',(df['floors'].unique())),
					'sqft' : st.slider('Luas Tanah.', 1000,max(df['sqft_living']),step=100),
					'waterfront':1 if st.checkbox('Tepi Laut.') else 0
					}
					df=df[df['bedrooms']==params['bedrooms']]
					df=df[df['bathrooms']==params['bathrooms']]
					df=df[df['floors']==params['floors']]
					df=df[df['waterfront']==params['waterfront']]
					df=df[(df['sqft_living']>0.9*params['sqft']) & (df['sqft_living']<1.1*params['sqft'])]
					df.reset_index()
					df['lat']=[get_locations(df.iloc[[i]]['zip'].values.astype(int))[0] for i in range(len(df))]
					df['lon']=[get_locations(df.iloc[[i]]['zip'].values.astype(int))[1] for i in range(len(df))]
					y=df['price']
					X=df[['bedrooms','bathrooms','floors','sqft_living','waterfront']]
					X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
					models = DecisionTreeRegressor(max_depth=25)
					df_models = pd.DataFrame()
					temp = {}
					print(models)
					m = str(models)
					temp['Model'] = m[:m.index('(')]
					models.fit(X_train, y_train)
					MAE = sqrt(mse(y_test, models.predict(X_test)))
					PRED =models.predict(pd.DataFrame(params,  index=[0]))[0]
					df_models = df_models.append(df)


					btn = st.button("Kalkulasi")
					if btn:
						st.write('Prediksi Harga Rumah Rata-Rata **${:.2f}**'.format(PRED))
						st.map(df_models[['lat','lon']])
						df_models
					else:
						pass

					st.subheader('Informasi Tambahan')
					if st.checkbox('Show ML MAE'):
						st.write('MAE Harga = **{:.2f}**'.format(MAE))
					if st.sidebar.button('Show JSON'):
						st.json(df[['price','bedrooms','bathrooms','floors','sqft_living','waterfront']].to_json())
					if st.sidebar.button('Close JSON'):
						df
				elif pilih == "Add User":
					new_username = st.text_input("User name")
					new_password = st.text_input("Password", type="password")
					confirm_password = st.text_input("Masukan Password Lagi", type='password')
					if new_password == confirm_password:
						st.success("Password cocok")
					else:
						st.warning("Password tidak cocok")
					if st.button("Submit"):
						create_usertable()
						hashed_new_password = generate_hash(new_password)
						add_userdata(new_username,hashed_new_password)
						st.success("Sucses Membuat akun")
						st.info("Silahkan Login Untuk Mencoba")
				elif pilih == "Lihat User":
					user_result = view_all_users()
					clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
					st.dataframe(clean_db)
			else:
				df=load_data()
				st.subheader('Property Options')

				params={
				'bedrooms' : st.selectbox('Kamar.',(df['bedrooms'].unique())),
				'bathrooms' : st.selectbox('Kamar Mandi.',(df['bathrooms'].unique())),
				'floors' : st.selectbox('Jumlah Lantai.',(df['floors'].unique())),
				'sqft' : st.slider('Luas Tanah.', 1000,max(df['sqft_living']),step=100),
				'waterfront':1 if st.checkbox('Tepi Laut.') else 0
				}
				df=df[df['bedrooms']==params['bedrooms']]
				df=df[df['bathrooms']==params['bathrooms']]
				df=df[df['floors']==params['floors']]
				df=df[df['waterfront']==params['waterfront']]
				df=df[(df['sqft_living']>0.9*params['sqft']) & (df['sqft_living']<1.1*params['sqft'])]
				df.reset_index()
				df['lat']=[get_locations(df.iloc[[i]]['zip'].values.astype(int))[0] for i in range(len(df))]
				df['lon']=[get_locations(df.iloc[[i]]['zip'].values.astype(int))[1] for i in range(len(df))]
				y=df['price']
				X=df[['bedrooms','bathrooms','floors','sqft_living','waterfront']]
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
				models = DecisionTreeRegressor(max_depth=25)
				df_models = pd.DataFrame()
				temp = {}
				print(models)
				m = str(models)
				temp['Model'] = m[:m.index('(')]
				models.fit(X_train, y_train)
				MAE = sqrt(mse(y_test, models.predict(X_test)))
				PRED =models.predict(pd.DataFrame(params,  index=[0]))[0]
				df_models = df_models.append(df)


				btn = st.button("Predict")
				if btn:
					st.write('Prediksi Harga Rumah Rata-Rata **${:.2f}**'.format(PRED))
					st.map(df_models[['lat','lon']])
					df_models
				else:
					pass
					
		else:
			st.warning("salah Password Bro")

if __name__ == '__main__':
	main()
