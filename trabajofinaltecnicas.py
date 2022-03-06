# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:28:25 2022
"""
#%%
#Comenzamos con la descarga de datos, usando la librería phd_scraper se obtuvo los datos de la estación del OVH almacenados en el servidor de SENAMHI, donde tuvimos que especificar el código de estación y las fechas
from phd_scraper import se_hydrometeo
se_hydrometeo.download(station_code='472AC278', init_date='2017-01-01', last_date='2021-12-31',to_csv='ovh_1721.csv')
#%%
#Leemos el csv y guardamos la columna de fechas que usaremos mas adelante
import pandas as pd
df=pd.read_csv('ovh_1721.csv')
fechas=df['DATE']
print(df)
#%%
#Descartamos las columnas de datos que no analizaremos
df=df.drop(['PREC_H','W_DIR','W_VEL'],axis=1)
print(df)
#%%
#"Desarmamos" la columna de fechas en columnas de años meses y días lo cual nos ayudará mas adelante para hacer cálculos más rápidos
df[['Y','M','D']]=df['DATE'].str.split('-',expand=True)
print(df)
#%%
#Reordenamos las columnas
df=df[['Y','M','D','HOUR','TEMP','HUM']]
print(df)
#%%
#Ploteamos la data descargada
import matplotlib.pyplot as plt
plt.figure()
plt.plot(pd.to_datetime(fechas),df.TEMP)
plt.ylabel('°C')
plt.title('Temperatura de la estación OVH')
plt.grid()

plt.figure()
plt.plot(pd.to_datetime(fechas),df.HUM)
plt.ylabel('%')
plt.title('Humedad Relativa de la estación OVH')
plt.grid()
#%%
#Se observa en los gráficos que hay valores outliers y al mismo tiempo data faltante, comenzaremos convirtiendo los outliers en nan, esto transformando valores mayores al percentil 99 en nan y los valores menores al percentil 1 en nan para ambas variables, al mismo tiempo observaremos cuantos valores nan había antes y despues de este proceso
print(df.isna().sum())
import numpy as np
t99=np.nanpercentile(df['TEMP'], 99)
t01=np.nanpercentile(df['TEMP'], 1)
h99=np.nanpercentile(df['HUM'], 99)
h01=np.nanpercentile(df['HUM'], 1)
for i in range(len(df)):
    if df['TEMP'][i]>t99:
        df['TEMP'][i]=np.nan
    elif df['TEMP'][i]<t01:
        df['TEMP'][i]=np.nan

for i in range(len(df)):
    if df['HUM'][i]>h99:
        df['HUM'][i]=np.nan
    elif df['HUM'][i]<h01:
        df['HUM'][i]=np.nan
print(df.isna().sum())
#%%
#En base a lo anterior haremos un completado de la data faltante horaria a través de un promedio mensual horario específico para cada mes de todos los años y tras esto ya no tendríamos valores nan como se imprime a continuación
prom_hm=df.groupby(['Y','M','HOUR'],as_index=False).mean()
ind_t=df.loc[pd.isna(df['TEMP']), :].index
for i in ind_t:
    cor=(int(df['Y'][i])-2017)*12*24+(int(df['M'][i])-1)*24+int(df['HOUR'][i][:2])
    df['TEMP'][i]=prom_hm['TEMP'][cor]
ind_hum=df.loc[pd.isna(df['HUM']), :].index
for i in ind_hum:
    cor=(int(df['Y'][i])-2017)*12*24+(int(df['M'][i])-1)*24+int(df['HOUR'][i][:2])
    df['HUM'][i]=prom_hm['HUM'][cor]
print(df.isna().sum())
#%%
#Convertimos la data horaria en data diaria
T_d=np.zeros(int(len(df)/24))
for i in range(int(len(df)/24)):
    T_d[i]=np.mean(df['TEMP'][i*24:i*24+24])
print(T_d)
H_d=np.zeros(int(len(df)/24))
for i in range(int(len(df)/24)):
    H_d[i]=np.mean(df['HUM'][i*24:i*24+24])
print(H_d)
#%%
#En base a lo que mencionamos al inicio usaremos la columna de fechas original y tomaremos los datos diarios
f_d=fechas.loc[0::24].reset_index(drop=True)
print(f_d)
#%%
#Creamos un dataframe final en donde guardaremos los datos que tenemos y pondremos las predicciones que usaremos, al mismo tiempo convertiremos en indice la columna de fechas
df_final=pd.DataFrame()
df_final['Fecha']=f_d
df_final['Fecha']=pd.to_datetime(df_final['Fecha'])
df_final['Temp']=T_d
df_final['HR']=H_d
df_final=df_final.set_index('Fecha')
print(df_final)

#%%
# Importamos las librerías para descompisición estacional para el análisis de la data y la librería de donde sacaremos los algoritmos para la predicción en este caso HoltWinters, y graficaremos la descomposición de los datos con un modelo multiplicativo
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
plt.figure()
decompose_result = seasonal_decompose(df_final['Temp'],model='multiplicative')
decompose_result.plot()
#%%
#Usaremosel modelo de suavizado exponencial simple de manera inicial
d=365
alpha = 1/(2*d)
df_final['HW1'] = SimpleExpSmoothing(df_final['Temp']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
df_final[['Temp','HW1']].plot(title='Suavizado exponencial simple- HoltWinter')
#%%
#Comparamos con el modelo exponencial doble de tendencia tanto aditiva como acumulativa sin tener muchas diferencias entre ellos pero al mismo tiempo siendo mucho mas precisos
df_final['HW2_ADD'] = ExponentialSmoothing(df_final['Temp'],trend='add').fit().fittedvalues
df_final['HW2_MUL'] = ExponentialSmoothing(df_final['Temp'],trend='mul').fit().fittedvalues

df_final[['Temp','HW2_ADD','HW2_MUL']].plot(title='Suavizado exponencial doble -HoltWinter:\n Tendencias Aditivas y Multiplicativas')
#%%
#Continuamos con un suavizado exponencial triple pero ahora agregamos una estacionalidad a los datos con frecuencia de 365 días para representar los años y una ciclidad en las tendencias para tener la tendencia mas precisa
df_final['HW3_ADD'] = ExponentialSmoothing(df_final['Temp'],trend='add',seasonal='add',seasonal_periods=365).fit().fittedvalues

df_final['HW3_MUL'] = ExponentialSmoothing(df_final['Temp'],trend='mul',seasonal='mul',seasonal_periods=365).fit().fittedvalues
df_final[['Temp','HW3_ADD','HW3_MUL']].plot(title='Suavizado exponencial triple - Holtwinter:\n Estacionalidades Aditivas y Multiplicativas')
#%%
#Ahora muy bien al ser el modelo desuavizado triple con tendencias estacionales el más preciso pasaremos a la parte de predicción donde tomaremos los primeros 4(2017-2020) años como data de entrenamiento para tener pronósticos para la data del 2021
entre_T=df_final[:365*4+1]
prueb_T=df_final[365*4+1:]

fitted_model = ExponentialSmoothing(entre_T['Temp'],trend='mul',seasonal='mul',seasonal_periods=365).fit()
predic_T = fitted_model.forecast(365)
plt.figure()
entre_T['Temp'].plot(legend=True,label='Entrenamiento')
prueb_T['Temp'].plot(legend=True,label='Prueba',figsize=(6,4))
predic_T.plot(legend=True,label='Predicción')
plt.title('Data de entrenamiento, prueba y predicción\n utilizando Holtwinter en Temperatura')
df_compar=pd.DataFrame()
df_compar['Temp']=prueb_T['Temp']
df_compar['Temp_Pred']=predic_T
#%%
#Realizamos el mismo proceso pero ahorapara humedad relativa
entre_T=df_final[:365*4+1]
prueb_T=df_final[365*4+1:]

fitted_model = ExponentialSmoothing(entre_T['HR'],trend='mul',seasonal='mul',seasonal_periods=365).fit()
predic_T = fitted_model.forecast(365)
plt.figure()
entre_T['HR'].plot(legend=True,label='Entrenamiento')
prueb_T['HR'].plot(legend=True,label='Prueba',figsize=(6,4))
predic_T.plot(legend=True,label='Predicción')
plt.title('Data de entrenamiento, prueba y predicción\nutilizando Holtwinter en Humedad Relativa')
df_compar['HR']=prueb_T['HR']
df_compar['HR_Pred']=predic_T
#%%
print(df_compar)
df_compar.to_csv('tabla_comparar.csv')
