#!/usr/bin/env python
# coding: utf-8

# <i style= 'color: green; font-size:1.5em'> Pamela Jaramillo
# <!---line--->

# # Ejercicio de machine learning: clasificación y regresión vinícola

# Dataset sobre distintos vinos con sus características (como pueden ser la acidez, densidad...). 
# 
# El dataset proviene de la Universdad de Minho, generado por [P. Cortez](http://www3.dsi.uminho.pt/pcortez/Home.html) et al. Dicho dataset se encuentra en el [*UC Irvine Machine Learning Repository*](https://archive.ics.uci.edu/ml/index.html) ([aquí](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) está disponible; pero debes usar la versión adjunta en la misma carpeta que este documento). Adjunto la descripción del dataset:
# 
# ```
# Citation Request:
#   This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
#   Please include this citation if you plan to use this database:
# 
#   P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#   Modeling wine preferences by data mining from physicochemical properties.
#   In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
# 
#   Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
#                 [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
#                 [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib
# 
# 1. Title: Wine Quality 
# 
# 2. Sources
#    Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
#    
# 3. Past Usage:
# 
#   P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#   Modeling wine preferences by data mining from physicochemical properties.
#   In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
# 
#   In the above reference, two datasets were created, using red and white wine samples.
#   The inputs include objective tests (e.g. PH values) and the output is based on sensory data
#   (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
#   between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
#   these datasets under a regression approach. The support vector machine model achieved the
#   best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
#   etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
#   analysis procedure).
#  
# 4. Relevant Information:
# 
#    The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
#    For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
#    Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
#    are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
#    These datasets can be viewed as classification or regression tasks.
#    The classes are ordered and not balanced (e.g. there are munch more normal wines than
#    excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
#    or poor wines. Also, we are not sure if all input variables are relevant. So
#    it could be interesting to test feature selection methods. 
# 
# 5. Number of Instances: red wine - 1599; white wine - 4898. 
# 
# 6. Number of Attributes: 11 + output attribute
#   
#    Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
#    feature selection.
# 
# 7. Attribute information:
# 
#    For more information, read [Cortez et al., 2009].
# 
#    Input variables (based on physicochemical tests):
#    1 - fixed acidity
#    2 - volatile acidity
#    3 - citric acid
#    4 - residual sugar
#    5 - chlorides
#    6 - free sulfur dioxide
#    7 - total sulfur dioxide
#    8 - density
#    9 - pH
#    10 - sulphates
#    11 - alcohol
#    Output variable (based on sensory data): 
#    12 - quality (score between 0 and 10)
# 
# 8. Missing Attribute Values: None
# ```

# Además de las 12 variables descritas, el dataset a utilizar tiene otra: si el vino es blanco o rojo. Dicho esto, los objetivos son:
# 
# 1. Separar el dataset en training y testing, con las transformaciones que se han considerado oportunas, así como selección de variables, reducción de dimensionalidad... 
# 2. Hacer un modelo capaz de clasificar lo mejor posible si un vino es blanco o rojo a partir del resto de variables.
# 3. Hacer un modelo regresor que prediga lo mejor posible la calidad de los vinos.
# 
# El fichero csv a utilizar `winequality.csv` tiene las cabeceras de cuál es cada variable, y los datos están separados por punto y coma.


# In[1]:


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from math import sqrt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

#______________MODELOS DE CLASIFICACIÓN__________________
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.neural_network import MLPClassifier as MLP


from sklearn.model_selection import GridSearchCV as GSC


#______________MÉTRICAS DE CLASIFICACIÓN__________________

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

############## MODELOS DE REGRESIÓN ##########################


from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LRR #para no confundirla con la LR de clasificación
from sklearn.linear_model import Ridge ## Regularización L2
from sklearn.linear_model import Lasso  ## Regularización L1
from sklearn.linear_model import ElasticNet ## Regularización L1 + L2
from sklearn.tree import DecisionTreeRegressor as DTR #árbol regresor
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.neural_network import MLPRegressor as MLPR

#______________MÉTRICAS DE REGRESIÓN_______________________


from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

#TRATO DE VARIABLES CATEGÓRICAS
from sklearn.preprocessing import LabelBinarizer
from sklearn.compose import make_column_transformer


#GRAFICOS
import seaborn as sns
import pandas_profiling
from prettytable import PrettyTable


#PIPELINE
from sklearn.pipeline import Pipeline

#FEATURE SELECTION
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
random_state=1987


# In[2]:


#FUNCIONES
def metricas(modelo,x,y):
    print('Accuracy: %s'%(round(((modelo.score(x,y))*100),2)))
    print('   Acierto un %s%% de veces'%(round(((modelo.score(x,y))*100),2)))
    print("Recall: %s" %(round(((recall_score(y_pred=modelo.predict(x),y_true =y))*100),2)))
    print("   Capto el %s%% de vino blanco" %(round(((recall_score(y_pred=modelo.predict(x),y_true =y))*100),2)))
    print("Precisión: %s" %(round(((precision_score(y_pred=modelo.predict(x),y_true =y))*100),2)))
    print("   Cuando predigo vino blanco lo hago con un %s%% de probabilidad" %(round(((precision_score(y_pred=modelo.predict(x),y_true =y))*100),2)))      
    print('\n')
    print('Matriz de confusión')
    confusion=confusion_matrix(y_true=y, y_pred=modelo.predict(x))
    print(confusion)
    print('Predigo %s veces que es blanco cuando es rojo y predigo %s veces que es rojo cuando es blanco' %(confusion[0,1],
                                                                                                       confusion[1,0]))
    
def prettytable(modelo1,modelo2,modelo3,x,y,lista):
    t = PrettyTable(lista)
    t.add_row(['Accuracy',round(((modelo1.score(x,y))*100),2),
           round(((modelo2.score(x,y))*100),2),
          round(((modelo3.score(x,y))*100),2)])
    t.add_row(['Recall', round(((recall_score(y_pred=modelo1.predict(x),y_true =y))*100),2),
          round(((recall_score(y_pred=modelo2.predict(x),y_true =y))*100),2),
          round(((recall_score(y_pred=modelo3.predict(x),y_true =y))*100),2)])
    t.add_row(['Precision', round(((precision_score(y_pred=modelo1.predict(x),y_true =y))*100),2),
          round(((precision_score(y_pred=modelo2.predict(x),y_true =y))*100),2),
          round(((precision_score(y_pred=modelo3.predict(x),y_true =y))*100),2)])
    return t
    
def report_met(modelo, x,y):
    y_true = y
    y_pred = modelo.predict(x)
    target_names = ['Vino Rojo','Vino Blanco']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
def metricas_reg(modelo,x,y):
    red=np.round(modelo.best_estimator_.predict(x),0)
    R2=round(r2(y_pred=red,y_true=y),3)
    MAE=round((mae(y_pred=red,y_true=y)),3)
    RMSE=round(np.sqrt(mse(y_pred=red,y_true=y)),3)
    print('R2:%s,MAE:%s,RMSE:%s'%(R2,MAE,RMSE))


# <a style='color:blue'> ANÁLISIS DE DATOS

# In[3]:


#DATAFRAME
df_wine1 = pd.read_csv ('winequality.csv',sep=';', delimiter=None, header=0)
df_wine1.head()


# In[4]:


#Reporte variables adjunto
report=pandas_profiling.ProfileReport(df_wine1)
report.to_file('profile_report.html')


# <a style= 'color:green'> 
# <!---blank line--->
# CONCLUSIONES
#  - En el reporte adjunto vemos que hay 1177 lineas duplicadas. Procederemos a eliminarlas para que no distorsionen la predicción.
#  - Encontramos también presencia de outliers en algunas variables.

# In[5]:


df_wine=df_wine1.copy()


# In[6]:


df_wine.drop_duplicates(inplace=True)


# In[7]:


report=pandas_profiling.ProfileReport(df_wine)
report.to_file('profile_report2.html')


# <a style='color:blue'> GRÁFICOS

# In[8]:


sns.pairplot(df_wine, 
             hue=("color"),
             vars=['fixed_acidity','volatile_acidity',
                   'citric_acid',
                   'residual_sugar',
                  'chlorides','free_sulfur_dioxide','total_sulfur_dioxide',
                  'density', 'pH','sulphates', 'alcohol', 'quality'],
            palette='Set1')
pass


# <a style='color:green'> Es complicado ver tendencias entre tantas variables, sin embargo se observa que la correlacion entre unas y otras variables también define el color del vino, por ejemplo vemos que:
#     - El vino blanco tiene menos acidez fija que el rojo
#     - Hay más presencia de sulfatos en vinos rojos que blancos.
# Veremos más adelante el mismo gráfico pero considerando solo aquellas variables que tienen correlación de media a alta con el
# color del vino

# ## <i style= 'color:red' > MODELOS DE CLASIFICACIÓN: PREDECIR EL COLOR DEL VINO

# In[9]:


palette ={'red':"C0",'white':"C3"}
sns.countplot(df_wine['color'],palette=palette)
pass


# In[10]:


df_wine.shape


# <a style= 'color:blue' >Transformamos la variable categórica a numérica:

# In[7]:


color_cat= df_wine['color'].copy()
encoder = LabelBinarizer()
color_cat_lb = encoder.fit_transform(color_cat)
color_cat_lb


# In[8]:


df_wine['color']=color_cat_lb


# In[9]:


df_wine.head()


# In[14]:


df_wine['color'].value_counts()


# In[15]:


df_wine.shape


# In[16]:


df_wine.describe()


# <a style= 'color:blue'> CORRELACIONES ENTRE VARIABLES
# 

# In[17]:


features=list(df_wine.columns)
features=features[0:12]


# In[18]:


plt.figure(figsize=(20,10))
sns.heatmap(df_wine.corr(), 
            vmin=-1.0,                       
            vmax=1.0,                       
            annot=True,                      
            cmap="coolwarm")                   
                                             
pass


# <a style= 'color:green' >Podemos observar que nuestra target tiene correlaciones altas con el sulfuro dióxido y con la acidez volátil, esta última es una correlación negativa.
# Tiene también correlaciones medias con la acidez fija, los cloruros y los sulfatos, con todas estas la relación es negativa también.
# <!----blank line---->
# Veremos en el siguiente gráfico las correlaciones de las variables con el color superiores al +-45% 

# In[19]:


df_wine_grafico=df_wine.copy()


# In[20]:


df_wine_grafico=df_wine_grafico.drop([
'citric_acid',
'residual_sugar',
'density',
'pH',
'alcohol',
'quality'], axis=1)


# In[21]:


df_wine_grafico.head()


# In[22]:


sns.pairplot(df_wine_grafico, 
             hue=("color"),
             vars=['fixed_acidity',
'volatile_acidity',
'chlorides',
'free_sulfur_dioxide',
'total_sulfur_dioxide',
'sulphates'
                  ],
             palette='Set1'
             
            )
pass


# <a style= 'color:green'> Con estas variables vemos mejor que existen patrones de separación claros entre el cruce de varias variables y el color del vino, como la acidez, los cloruros y el sulfuro dioxido

# <a style='color:blue'> CONJUNTO DE TEST Y TRAIN

# In[23]:


df_wine_train, df_wine_test = train_test_split(df_wine,
                                    train_size=0.7,
                                              random_state=1987)


# In[24]:


print('Train:\n %s'%df_wine_train['color'].value_counts())
print('Test:\n %s'%df_wine_test['color'].value_counts())


# <a style='color:green'> En la muestra de datos tenemos más cantidad de vinos rojos que blancos, la proporción para el conjunto
#     de train y test se mantiene según la muestra original. Veremos si el modelo es mejor prediciendo un color u otro.

# <a style='color:blue'> ESCALANDO LOS DATOS

# <a style='color:green'> En el análisis de datos hemos detectado outliers en algunas variables, por lo que utilizaremos el metodo Standardscaler que es menos sensible a outliers

# In[25]:


#Solo escalaremos las features. No la variable target
df_wine_train2=df_wine_train.iloc[:,0:12].copy()


# In[26]:


df_wine_train2.head()


# In[27]:


#Entrenamos con el conjunto de train
scaler = StandardScaler()
scaler.fit(df_wine_train2)


# In[28]:


df_wine_train_sc = scaler.transform(df_wine_train2)
df_wine_train_sc


# In[29]:


df_wine_test2=df_wine_test.iloc[:,0:12].copy()
df_wine_test_sc = scaler.transform(df_wine_test2)
df_wine_test_sc


# In[30]:


columnas=list(df_wine.columns)
columnas2=columnas[0:12]


# In[31]:


df_wine_train_sc=pd.DataFrame(df_wine_train_sc, columns=columnas2)
df_wine_test_sc=pd.DataFrame(df_wine_test_sc, columns=columnas2)


# In[32]:


features=list(df_wine.columns)
features=features[0:12]
features


# In[33]:


#Conjuntos estandarizados
x_train=df_wine_train_sc[features]
y_train=df_wine_train["color"]

x_test=df_wine_test_sc[features]
y_test=df_wine_test["color"]


# In[34]:


y_test.value_counts()


# In[35]:


y_train.value_counts()


# In[36]:


#Conjuntos no estandarizados
x_train2=df_wine_train[features]
y_train2=df_wine_train["color"]

x_test2=df_wine_test[features]
y_test2=df_wine_test["color"]


# <a style='color:green'> Probaremos los modelos en datos estandarizados y sin estandarizar, para comprobar con qué tipo de datos obtenemos mejores resultados

# <a style='color:blue; font-size:1.5em'> MODELOS

# <a style='color:blue'> MODELO 1: GRID SEARCH DE REGRESIÓN LOGARITMICA 
# 

# In[37]:


#Datos estandarizados
gs_lr=GSC(estimator=LR(),
          param_grid={'C':[1,0.5],
                    'fit_intercept':[True,False],
                      'random_state':[1987]
                     },
          cv=5,
          verbose=2)
#Datos sin estandarizar
gs_lr2=GSC(estimator=LR(),
          param_grid={'C':[1,0.5],
                    'fit_intercept':[True,False],
                      'random_state':[1987]
                     },
          cv=5,
          verbose=2)


# In[38]:


#Entrenando el modelo 
gs_lr.fit(x_train,y_train)
gs_lr2.fit(x_train2,y_train2)


# In[40]:


best_lr=gs_lr.best_estimator_
best_lr2=gs_lr2.best_estimator_


# <a style='color:blue'> MÉTRICAS

# In[41]:


print('Estandarizado')
print(metricas(best_lr,x_test,y_test))
print('\n')
print('No Estandarizado')
print(metricas(best_lr2,x_test2,y_test2))


# In[42]:


print('Estandarizado')
print(report_met(best_lr,x_test,y_test))
print('\n')
print('No Estandarizado')
print(report_met(best_lr2,x_test2,y_test2))


# <a style='color:green'>CONCLUSION 1: Modelo de regresión logarítmica presenta mejores resultados con datos estandarizados, aunque la diferencia es mínima. Vemos en la tabla superior que cuando se trata de predecir vino blanco en el modelo estandarizado acierto el 100% de las veces y cuando se trata de predecir vino rojo acierto el 98% de las veces. 
# El hecho de tener menos muestras para un tipo de vino no afecta significativamente en la predicción en este caso.

# <a style='color:blue'> MODELO 2: GRID SEARCH DECISION TREE CLASSIFIER

# In[43]:


#ESTANDARIZADOS
gs_dtc=GSC(estimator=DTC(),
          param_grid={'criterion':['gini','entropy'],
                    'max_depth':[2,3,4,5,6],
                      'random_state':[1987]
                     },
          cv=5,
          verbose=0)
#SIN ESTANDARIZAR
gs_dtc2=GSC(estimator=DTC(),
          param_grid={'criterion':['gini','entropy'],
                    'max_depth':[2,3,4,5,6],
                      'random_state':[1987]
                     },
          cv=5,
          verbose=0)


# In[44]:


gs_dtc.fit(x_train,y_train)
gs_dtc2.fit(x_train2,y_train2)


# In[45]:


best_dtc=gs_dtc.best_estimator_
best_dtc2=gs_dtc2.best_estimator_


# In[46]:


print('Estandarizado')
print(metricas(best_dtc,x_test,y_test))
print('\n')
print('No Estandarizado')
print(metricas(best_dtc2,x_test2,y_test2))


# In[47]:


print('Estandarizado')
print(report_met(best_dtc,x_test,y_test))
print('\n')
print('No Estandarizado')
print(report_met(best_dtc2,x_test2,y_test2))


# <a style='color:green '>CONCLUSION: La diferencia en un modelo de arbol entre datos estandarizados y sin estandarizar es más baja que en la regresión logística. Este modelo tiene peores métricas que la regresión logística.

# <a style='color:blue'> SIGNIFICANCIA DE LAS VARIABLES

# In[48]:


#La significancia de las variables la calculo sobreestimando el modelo
dtc_importance=DTC(max_depth=10, random_state=1987)
dtc_importance.fit(df_wine_train[features], df_wine_train['color'])
variables=dtc_importance.feature_importances_


# In[49]:


df_featureimportances=pd.DataFrame(variables, columns=['Importancia'])
df_featureimportances['Variable']=features
df_featureimportances


# <a style='color:green'>El sulfuro dioxido y los cloruros son los que más aportan para la predicción del color entre las dos suman un 87,9%. Podríamos usar solo esas variables para obtener buenos resultados. Ya nos lo indicaba así nuestra matriz de correlaciones
# en las que se veía una alta correlación del color con estas dos variables.
# <!---blank line--->
# Los cloruros son la clave para determinar el color del vino. junto con el sulfuro dioxido explican el color del vino en un 87.9%
# <!---blank line--->
# En este caso al ser pocas las variables involucradas para la predicción, decidimos usar todas, como obtenemos modelos realmente buenos no hace falta quitar ninguna variable que pueda estar empeorando el modelo.
# 

# In[50]:


sns.scatterplot(x='total_sulfur_dioxide', y='chlorides', hue='color', data=df_wine1)
pass


# <a style= 'color:green'>
# Según el gráfico, la mayoría de muestras de vino blanco tiene cantidades de cloruro más bajas que el vino rojo y el vino
# blanco tiene cantidades de sulfuro dioxido más altas  que el vino rojo.

# <a style='color:blue'> MODELO 3: GRID SEARCH CON RANDOM FOREST CLASSIFIER

# In[51]:


#DATOS ESTANDARIZADOS
gs_rfc=GSC(estimator=RFC(),
          param_grid={'criterion':['gini','entropy'],
                    'max_depth':[2,3,4,5,6,7,8,9],
                      'n_estimators':[5,10,15,20,25],
                      'random_state':[1987]
                     },
          cv=5,
          verbose=0)
#DATOS SIN ESNTANDARIZAR
gs_rfc2=GSC(estimator=RFC(),
          param_grid={'criterion':['gini','entropy'],
                    'max_depth':[2,3,4,5,6,7,8,9],
                      'n_estimators':[5,10,15,20,25],
                      'random_state':[1987]
                     },
          cv=5,
          verbose=0)


# In[52]:


gs_rfc.fit(x_train,y_train)
gs_rfc2.fit(x_train2,y_train2)


# In[53]:


best_rfc=gs_rfc.best_estimator_
best_rfc2=gs_rfc2.best_estimator_


# In[54]:


print('Estandarizado')
print(metricas(best_rfc,x_test,y_test))
print('\n')
print('No Estandarizado')
print(metricas(best_rfc2,x_test2,y_test2))


# In[55]:


print('Estandarizado')
print(report_met(best_rfc,x_test,y_test))
print('\n')
print('No Estandarizado')
print(report_met(best_rfc2,x_test2,y_test2))


# In[56]:


prettytable(best_lr,best_dtc,best_rfc,x_test,y_test,lista=['Métricas','Reg.Logística','Arbol Decisión','Random Forest'])


# <a style= 'color:green'> CONCLUSION: Modelo ganador: RANDOM FOREST CLASSIFIER en datos estandarizados.
# Todos los modelos dan métricas realmente buenas. RFC es más robusto por muy poca diferencia, como se puede ver en la tabla anterior. No hemos probado más modelos ya que hemos conseguido excelentes resultados con estos 3.

# ## <i style= 'color:red' > MODELOS DE REGRESIÓN: MODELOS PREDICCIÓN CALIDAD DEL VINO

# <a style='color:blue'> CONJUNTOS TRAIN Y TEST

# In[61]:


df_wine_train_quality=df_wine_train.copy()
df_wine_test_quality=df_wine_test.copy()


# In[62]:


df_wine_train_quality2=df_wine_train_quality.copy()
df_wine_test_quality2=df_wine_test_quality.copy()

df_wine_train_quality2=df_wine_train_quality2.drop('quality', axis=1)
df_wine_test_quality2=df_wine_test_quality2.drop('quality', axis=1)


# In[63]:


x_train_q=df_wine_train_quality2
y_train_q=df_wine_train_quality['quality']

x_test_q=df_wine_test_quality2
y_test_q=df_wine_test_quality['quality']


# <a style='color:blue'> PIPELINES

# In[64]:



#RANDOM FOREST REGRESOR WITH SELECTKBEST
rfr_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("rfr",RFR())
                              ]
                       )
#RANDOM FOREST REGRESOR WITH RFECV

rfr_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("rfr",RFR())
                              ]
                       )

#LINEAR REGRESION REGRESOR WITH SELECTKBEST
lrr_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("lrr",LRR())
                              ]
                       )
#LINEAR REGRESION REGRESOR WITH RFECV
lrr_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("lrr",LRR())
                              ]
                       )

#RIDGE  WITH RFECV
ridge_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("ridge",Ridge())
                              ]
                       )
#RIDGE  WITH SELECTKBEST
ridge_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("ridge",Ridge())
                              ]
                       )

#LASSO  WITH SELECTKBEST
lasso_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("lasso",Lasso())
                              ]
                       )
#LASSO  WITH RFECV
lasso_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("lasso",Lasso())
                              ]
                       )
#ELASTIC NET  WITH RFECV
elastic_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("elastic",ElasticNet())
                              ]
                       )

#ELASTIC NET  WITH SELECTKBEST
elastic_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("elastic",ElasticNet())
                              ]
                       )
#DECISION TREE REG  WITH SELECTKBEST
dtr_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("dtr",DTR())
                              ]
                       )
#DECISION TREE REG  WITH RFECV
dtr_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("dtr",DTR())
                              ]
                       )
#ADABOOSTER  WITH RFECV
abr_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("abr",ABR())
                              ]
                       )
#ADA BOOST  WITH SELECTKBEST
abr_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("abr",ABR())
                              ]
                       )

#GRADIENT BOOST  WITH SELECTKBEST
gbr_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("gbr",GBR())
                              ]
                       )
#GRADIENT BOOST  WITH RFECV


#MLP REG  WITH RFECV

mlp_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("mlp",MLPR())
                              ]
                       )
#MLP REG  WITH SELECTKBEST

mlp_kbest = Pipeline(steps=[("scaler",StandardScaler()),
                               ("kbest",SelectKBest()),
                               ("mlp",MLPR())
                              ]
                       )



# In[79]:


#HIPERPARAMETROS


grid_rfr_kbest={'kbest__score_func': [f_classif],
                'rfr__criterion':['mae','mse'],
               'rfr__max_depth':[2,3,4,5,6,7,8,9,10],
               'rfr__n_estimators':[5,10,15,20,30,40],
               'rfr__random_state':[1987]}
grid_rfr_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'rfr__criterion':['mae','mse'],
               'rfr__max_depth':[2,3,4,5,6,7,8,9,10],
               'rfr__n_estimators':[5,10,15,20,30,40],
               'rfr__random_state':[1987]}
grid_lrr_kbest={'kbest__score_func': [f_classif],
                'lrr__fit_intercept':[True,False]}
grid_lrr_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'lrr__fit_intercept':[True,False]}
grid_ridge_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'ridge__solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                 'ridge__fit_intercept':[True,False],
               'ridge__random_state':[1987]}
grid_ridge_kbest={'kbest__score_func': [f_classif],
                'ridge__solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                 'ridge__fit_intercept':[True,False],
               'ridge__random_state':[1987]}
grid_lasso_kbest={'kbest__score_func': [f_classif],
                'lasso__fit_intercept':[True,False],
               'lasso__random_state':[1987]}
grid_lasso_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'lasso__fit_intercept':[True,False],
               'lasso__random_state':[1987]}
grid_elastic_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'elastic__l1_ratio':[0,1,0.5],
                   'elastic__fit_intercept':[True,False],
               'elastic__random_state':[1987]}
grid_elastic_kbest={'kbest__score_func': [f_classif],
                    'elastic__l1_ratio':[0,1,0.5],
                   'elastic__fit_intercept':[True,False],
               'elastic__random_state':[1987]}
grid_dtr_kbest={'kbest__score_func': [f_classif],
                'dtr__criterion':['mae','mse'],
               'dtr__max_depth':[2,3,4,5,6,7,8,9,10],
               'dtr__random_state':[1987]}
grid_dtr_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'dtr__criterion':['mae','mse'],
               'dtr__max_depth':[2,3,4,5,6,7,8,9,10],
               'dtr__random_state':[1987]}
grid_abr_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'abr__loss':['linear','square', 'exponential'],
                'abr__n_estimators':[10,20,30,50,60],
                'abr__random_state':[1987]}
grid_abr_kbest={'kbest__score_func': [f_classif],
                'abr__loss':['linear','square', 'exponential'],
                'abr__n_estimators':[10,20,30,50,60],
                'abr__random_state':[1987]}
grid_gbr_kbest={'kbest__score_func': [f_classif],
                'gbr__loss':['ls', 'lad', 'huber', 'quantile'],
                'gbr__n_estimators':[50,100,150,200],
                'gbr__criterion':['friedman_mse','mse','mae'],
                'gbr__max_depth':[2,3,4,5,6,7,8,9,10],
                'gbr__random_state':[1987]}

grid_mlp_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'mlp__activation':['identity', 'logistic', 'tanh', 'relu'],
               'mlp__solver':['lbfgs','sgd', 'adam'],
               'mlp__learning_rate':['constant', 'invscaling', 'adaptive'],
               'mlp__random_state':[1987]}
grid_mlp_kbest={'kbest__score_func': [f_classif],
                'mlp__activation':['identity', 'logistic', 'tanh', 'relu'],
               'mlp__solver':['lbfgs','sgd', 'adam'],
               'mlp__learning_rate':['constant', 'invscaling', 'adaptive'],
               'mlp__random_state':[1987]}





# In[179]:


gs_rfr_kbest=GSC(rfr_kbest,
                grid_rfr_kbest,
                cv=10)
gs_rfr_rfecv=GSC(rfr_rfecv,
                grid_rfr_rfecv,
                cv=10)
gs_lrr_kbest=GSC(lrr_kbest,
                grid_lrr_kbest,
                cv=10)
gs_lrr_rfecv=GSC(lrr_rfecv,
                grid_lrr_rfecv,
                cv=10)
gs_ridge_rfecv=GSC(ridge_rfecv,
                  grid_ridge_rfecv,
                  cv=10)
gs_ridge_kbest=GSC(ridge_kbest,
                  grid_ridge_kbest,
                  cv=10)
gs_lasso_kbest=GSC(lasso_kbest,
                  grid_lasso_kbest,
                  cv=10)
gs_lasso_rfecv=GSC(lasso_rfecv,
                  grid_lasso_rfecv,
                  cv=10)
gs_elastic_rfecv=GSC(elastic_rfecv,
                    grid_elastic_rfecv,
                    cv=10)
gs_elastic_kbest=GSC(elastic_kbest,
                    grid_elastic_kbest,
                    cv=10)
gs_dtr_kbest=GSC(dtr_kbest,
                grid_dtr_kbest,
                cv=10)
gs_dtr_rfecv=GSC(dtr_rfecv,
                grid_dtr_rfecv,
                cv=10)
gs_abr_rfecv=GSC(abr_rfecv,
                grid_abr_rfecv,
                cv=10)
gs_abr_kbest=GSC(abr_kbest,
                grid_abr_kbest,
                cv=10)
gs_gbr_kbest=GSC(gbr_kbest,
                grid_gbr_kbest,
                cv=10)

gs_mlp_rfecv=GSC(mlp_rfecv,
                grid_mlp_rfecv,
                cv=10,
                verbose=2)
gs_mlp_kbest=GSC(mlp_kbest,
                grid_mlp_kbest,
                cv=10,
                verbose=2)


pipelines2 = [gs_rfr_kbest,
gs_rfr_rfecv,
gs_lrr_kbest,
gs_lrr_rfecv,
gs_ridge_rfecv,
gs_ridge_kbest,
gs_lasso_kbest,
gs_lasso_rfecv,
gs_elastic_rfecv,
gs_elastic_kbest,
gs_dtr_kbest,
gs_dtr_rfecv,
gs_abr_rfecv,
gs_abr_kbest,
gs_gbr_kbest,
gs_gbr_rfecv,
gs_mlp_rfecv,
gs_mlp_kbest]


pipelines4 = [gs_rfr_kbest,
gs_rfr_rfecv,
gs_lrr_kbest,
gs_lrr_rfecv,
gs_ridge_rfecv,
gs_ridge_kbest,
gs_lasso_kbest,
gs_lasso_rfecv,
gs_elastic_rfecv,
gs_elastic_kbest,
gs_dtr_kbest,
gs_dtr_rfecv,
gs_abr_rfecv,
gs_abr_kbest,
gs_gbr_kbest]


# In[91]:


pipelines4 = [gs_rfr_kbest,
gs_rfr_rfecv,
gs_lrr_kbest,
gs_lrr_rfecv,
gs_ridge_rfecv,
gs_ridge_kbest,
gs_lasso_kbest,
gs_lasso_rfecv,
gs_elastic_rfecv,
gs_elastic_kbest,
gs_dtr_kbest,
gs_dtr_rfecv,
gs_abr_rfecv,
gs_abr_kbest,
gs_gbr_kbest]


# In[101]:


pipe_dict3 = {0:'Random FR Kbest',
1:'Random FR rfecv',
2:'Linear Regresion kbest',
3:'Linear Regresion rfecv',
4:'Ridge rfecv',
5:'Ridge kbest',
6:'Lasso kbest',
7:'Lasso rfecv',
8:'ElasticNet rfecv',
9:'ElasticNet kbest',
10:'Decision Tree kbest',
11:'Decision Tree rfecv',
12:'Ada Boost rfecv',
13:'Ada Boost kbest',
14:'Gradient Boost kbest'}


# <a style='color:green'> Hemos hecho pipelines que escalen los datos con StandardScaler, que seleccionen features con dos tipos de seleccionadores: Kbest y RFECV y además se han incluido 14 modelos distintos cada uno usando Grid search con distintos hiperparámetros

# In[ ]:


for pipe in pipelines2:
    print('ejecutando %s'%pipe)
    pipe.fit(x_train_q, y_train_q)


# <a style='color:blue'> MÉTRICAS

# In[178]:


for idx, val in enumerate(pipelines4):
    print(pipe_dict3[idx])
    print(metricas_reg(val,x_test_q,y_test_q))


# In[173]:


R2MAX=0
maemin=1000
rmsemin=1000
for idx, val in enumerate(pipelines4):
    R2=round(r2(y_pred=np.round(val.best_estimator_.predict(x_test_q),0),y_true=y_test_q),3)
    if R2>R2MAX:
        R2MAX=R2
        best_model=pipe_dict3[idx]
        
    MAE=round((mae(y_pred=np.round(val.best_estimator_.predict(x_test_q),0),y_true=y_test_q)),3)
    if MAE<maemin:
        maemin=MAE
        best_model_mae= pipe_dict3[idx]
    
    RMSE=round(np.sqrt(mse(y_pred=np.round(val.best_estimator_.predict(x_test_q),0),y_true=y_test_q)),3)
    if RMSE<rmsemin:
        rmsemin=RMSE
        best_model_rmse= pipe_dict3[idx]

print('Mejor modelo: %s R2:%s'%(best_model, R2MAX))
print('Mejor modelo: %s MAE:%s'%(best_model_mae,maemin))
print('Mejor modelo: %s RMSE:%s'%(best_model_rmse,rmsemin))


# In[175]:


t2 = PrettyTable(['Métricas','Modelo','Valor'])
t2.add_row(['RMSE',best_model_rmse,rmsemin])
t2.add_row(['MAE', best_model_mae,maemin])
t2.add_row(['R2', best_model,R2MAX])
print(t2)


# <a style='color:green'> Tras correr todos los pipelines el modelo más robusto es el Gradient Boost con selector de variables Kbest. Se han obtenido las siguientes métricas:
#     - RMSE: 0.742
#     - MAE: 0.465
#     Estos dos valores nos explican que la mayoria de veces nos equivocamos en media 0.74 puntos en predecir la calidad del vino
#     - R2: 0.26
#     El R2 nos explica que la calidad del vino está explicada en un 26% por las variables escogidas.
# Esto nos muestra que con estas variables es complicado predecir la calidad de forma perfecta, no son variables que tengan correlaciones altas con la calidad como se aprecia en el siguiente gráfico. Tiene correlaciones inferiores al 10% con el 58% de variables, correlaciones entre 11% y 33% con el 25% de las variables y solamente con el alcohol tiene una correlación del 47%, esto explica perfectamente el bajo valor de la última métrica, que permite concluir que con estas variables es complicado predecir la calidad del vino de manera precisa. 

# ![alt text](captura.jpg "Title")

# In[185]:


#probamos GBR con las mismas especificaciones que el best model pero con select rfecv
gbr_rfecv = Pipeline(steps=[("scaler",StandardScaler()),
                               ("rfecv",RFECV(estimator=LR())),
                               ("gbr",GBR())
                              ]
                       )
grid_gbr_rfecv={'rfecv__step': [1], 
                'rfecv__cv': [5],
                'gbr__loss':['huber'],
                'gbr__n_estimators':[150],
                'gbr__criterion':['friedman_mse'],
                'gbr__max_depth':[3],
                'gbr__random_state':[1987]}

gs_gbr_rfecv=GSC(gbr_rfecv,
                grid_gbr_rfecv,
                cv=10,
                verbose=2)


# In[186]:


gs_gbr_rfecv.fit(x_train_q,y_train_q)


# In[187]:


metricas_reg(gs_gbr_rfecv,x_test_q,y_test_q)
#no mejora con el feature selector rfecv


# In[180]:


#probamos mlps para ver si mejoran al gradient boost.
gs_mlp_rfecv.fit(x_train_q,y_train_q)


# In[188]:


gs_mlp_kbest.fit(x_train_q,y_train_q)


# In[184]:


metricas_reg(gs_mlp_rfecv,x_test_q,y_test_q)


# In[189]:


metricas_reg(gs_mlp_kbest,x_test_q,y_test_q)


# <a style='color:green'> Se probaron también MLPs pero no mejoran las métricas del Gradient Boost.
# El Gradient Boost es un modelo centrado en reducir el error, los hiperparámetros que mejor funcionaron son:
#     - Criterio de reduccion de error: Friedman-mse
#     - Learning rate=0.1
#     - Profundidad de los arboles=3
#     - Número de árboles=150
# El modelo se ha predicho y medido con los valores redondeados, es decir como los valores a predecir no pueden tener decimales, todos los resultados predichos son enteros, esto causa que el error disminuya.

# In[165]:


#detalles
mejor_pipeline = gs_gbr_kbest.best_estimator_

mejor_pipeline.steps


# In[174]:


#guardamos el modelo ganador
with open("mejor_pipeline_vino.model", "wb") as archivo_salida:
    pickle.dump(mejor_pipeline, archivo_salida)


# In[167]:


with open("mejor_pipeline_vino.model", "rb") as archivo_entrada:
    pipeline_importada = pickle.load(archivo_entrada)


# In[168]:


pipeline_importada


# <a style='color:green'> Como hemos visto, las métricas que nos da el modelo más robusto, a pesar de probar muchos, no son excelentes. Por eso procederemos a comprobar cómo está prediciendo realmente el modelo para ver con cuanto se equivoca realmente y en qué porcentaje.

# In[195]:


prediccion=pipeline_importada.predict(x_test_q)
prediccion=np.round(prediccion)
prediccion


# In[206]:


borrar=df_wine_test_quality.columns[0:11]


# In[228]:


df_wine_comparativa=df_wine_test_quality.drop(borrar,axis=1).copy()
df_wine_comparativa=df_wine_comparativa.drop('color',axis=1)
df_wine_comparativa['prediccion']=prediccion
df_wine_comparativa['diferencia']=df_wine_comparativa['prediccion']-df_wine_comparativa['quality']
df_wine_comparativa['error']=((df_wine_comparativa['prediccion']-
                                    df_wine_comparativa['quality'])**2)
df_wine_comparativa.head()


# In[224]:


#comprobamos el rmse que obtuvimos en el modelo
mse=np.sum(df_wine_comparativa['error'])/len(df_wine_comparativa)
rmse=round(np.sqrt(mse),3)
rmse


# In[263]:


#creamos un dataset con dos columnas nuevas, diferencia (que es la resta entre el valor real y el valor predicho) y
#error(que es el error cuadrático de la diferencia entre el valor real y el valor predicho)
df_wine_comparativa.describe()


# <a style='color:green'> Como vemos en el cuadro anterior, la media de la diferencia es de 0,06, es bastante baja, pero vemos también que el valor máximo es de 3 y el mínimo es -3. Es decir hay veces en las que nos equivocamos con + - 3 puntos en la predicción de la calidad. Vemos a continuación la incidiencia de este error

# In[262]:


df3 = df_wine_comparativa[(df_wine_comparativa['diferencia'] == 3) | (df_wine_comparativa['diferencia'] == -3)]
df2 = df_wine_comparativa[(df_wine_comparativa['diferencia'] == 2) | (df_wine_comparativa['diferencia'] == -2)]
df1 = df_wine_comparativa[(df_wine_comparativa['diferencia'] == 1) | (df_wine_comparativa['diferencia'] ==-1)]
df0 = df_wine_comparativa[df_wine_comparativa['diferencia'] == 0]

print('Error en 3/-3: %s'%(round(len(df3)/len(df_wine_comparativa),3)))
print('Error en 2/-2: %s'%(round(len(df2)/len(df_wine_comparativa),2)))
print('Error en 1/-1: %s'%(round(len(df1)/len(df_wine_comparativa),2)))
print('Sin error:%s'%(round(len(df0)/len(df_wine_comparativa),2)))


# <a style= 'color:blue'> CONCLUSIÓN 
# 

# <a style= 'color:green'> 
# El modelo acierta un 57% de las veces en la predicción de la calidad y el 39% se equivoca con +-1 punto. 
# Se equivoca con 2 puntos un 3% de las veces y el máximo error es de 3 puntos, equivocandose solo un 0.3% de veces.
#     
# No es un modelo totalmente preciso, como ya se ha comentado las variables no son las mejores para predecir la calidad, pero se puede concluir que 96 de cada 100 veces, acierta un 60% y un 40% se equivoca máximo con + - 1 punto en predecir la calidad. Es decir que es un modelo con un 96% de aciertos con tolerancia de +1-1 punto en la predicción.
# 
# 

# In[ ]:




