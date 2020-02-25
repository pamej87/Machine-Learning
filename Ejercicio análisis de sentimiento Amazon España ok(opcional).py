#!/usr/bin/env python
# coding: utf-8

# 
# <i style= 'color: green; font-size:1.5em'> Pamela Jaramillo
# <!---line--->
# <i style= 'color: green; font-size:1em'> Práctica 4

# # Análisis de sentimientos con reviews de productos de Amazon España (opcional)

# Si has hecho ya el ejercicio de web scraping con `Requests` y `BeautifulSoup` habrás visto cómo extraer datos de una página web.
# 
# El dataset que utilizarás en este ejercicio (que no es obligatorio entregar) lo he generado utilizando `Scrapy` y `BeautifulSoup`, y contiene unas $700.000$ entradas con dos columnas: el número de estrellas dadas por un usuario a un determinado producto y el comentario sobre dicho producto; exactamente igual que en el ejercico de scraping.
# 
# Ahora, tu objetivo es utilizar técnicas de procesamiento de lenguaje natural para hacer un clasificador que sea capaz de distinguir (¡y predecir!) si un comentario es positivo o negativo.
# 
# Es un ejercicio MUY complicado, más que nada porque versa sobre técnicas que no hemos visto en clase. Así que si quieres resolverlo, te va a tocar estudiar y *buscar por tu cuenta*; exactamente igual que como sería en un puesto de trabajo. Dicho esto, daré un par de pistas:
# 
# + El número de estrellas que un usuario da a un producto es el indicador de si a dicho usuario le ha gustado el producto o no. Una persona que da 5 estrellas (el máximo) a un producto probablemente esté contento con él, y el comentario será por tanto positivo; mientras que cuando una persona da 1 estrella a un producto es porque no está satisfecha... 
# + Teniendo el número de estrellas podríamos resolver el problema como si fuera de regresión; pero vamos a establecer una regla para convertirlo en problema de clasificación: *si una review tiene 4 o más estrellas, se trata de una review positiva; y será negativa si tiene menos de 4 estrellas*. Así que probablemente te toque transformar el número de estrellas en otra variable que sea *comentario positivo/negativo*.
# 
# Y... poco más. Lo más complicado será convertir el texto de cada review en algo que un clasificador pueda utilizar y entender (puesto que los modelos no entienden de palabras, sino de números). Aquí es donde te toca investigar las técnicas para hacerlo. El ejercicio se puede conseguir hacer, y obtener buenos resultados, utilizando únicamente Numpy, pandas y Scikit-Learn; pero siéntete libre de utilizar las bibliotecas que quieras.
# 
# Ahora escribiré una serie de *keywords* que probablemente te ayuden a saber qué buscar:
# 
# `bag of words, tokenizer, tf, idf, tf-idf, sklearn.feature_extraction, scipy.sparse, NLTK (opcional), stemmer, lemmatizer, stop-words removal, bigrams, trigrams`
# 
# No te desesperes si te encuentras muy perdido/a y no consigues sacar nada. Tras la fecha de entrega os daré un ejemplo de solución explicado con todo el detalle posible.
# 
# ¡Ánimo y buena suerte!

# In[2]:


import pandas as pd
import numpy as np
import re
import nltk 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle 

#-------REPRESENT TEXT TO NUMERICAL WORDS--------------
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#-------TRAIN TEST SPLIT--------------

from sklearn.model_selection import train_test_split

#-------MODELOS DE CLASIFICACIÓN--------------

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV as GSC
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import LinearSVC as SVC

#-------MÉTRICAS DE CLASIFICACIÓN--------------

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler
#PIPELINE
from sklearn.pipeline import Pipeline
#FEATURES SELECTION
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest as SKB


# In[3]:


#FUNCIONES
def metricas(modelo,x,y):
    Accuracy=(round(((modelo.score(x,y))*100),2))
    Recall=round(((recall_score(y_pred=modelo.predict(x),y_true =y))*100),2)
    precision=round(((precision_score(y_pred=modelo.predict(x),y_true =y))*100),2)
    print('Accuracy:%s'%Accuracy)
    print('   Acierto un %s%% de veces'%(Accuracy))
    print('Recall:%s'%Recall)
    print("   Capto el %s%% de comentarios positivos" %(Recall))
    print('Precisión:%s'%precision)
    print("   Cuando predigo un comentario positivo lo hago con un %s%% de fiabilidad" %(precision))      
    print('\n')
    print('Matriz de confusión')
    confusion=confusion_matrix(y_true=y, y_pred=modelo.predict(x))
    return confusion
    print('Predigo %s veces que es un comentario positivo cuando es negativo y predigo %s veces que es negativo cuando es positivo' %(confusion[0,1],
                                                                                                       confusion[1,0]))


# In[2]:


df_amazon= pd.read_csv('amazon_es_reviews/amazon_es_reviews.csv', sep=';', header=0,encoding='ISO-8859-1')


# In[3]:


df_amazon.head(20)


# <a style='color:blue'> Creación de una nueva variable, según número de estrellas

# In[4]:


valoracion=[]
for x in df_amazon['estrellas']:
    if x >=4:
        n=1
    else:
        n=0
        
    valoracion.append(n)
#1: comentarios positivos
#0: comentarios negativos


# In[5]:


df_amazon['valoracion']=valoracion


# In[6]:


df_amazon


# In[5]:


features = df_amazon.iloc[:, 0].values
target = df_amazon.iloc[:, 2].values


# In[4]:


len(features)


# In[6]:


df_amazon.head()


# <a style='color:blue'> Limpieza de comentarios

# In[9]:


#Limpiamos los comentarios quitando caracteres que puedan afectar a la predicción
features_limpias2=[]
for sentence in range(0, len(features)):
    features_limpias = re.sub('Ã¡','a', str(features[sentence])) #sustituir caracteres (vocales con tilde) por vocales 
    features_limpias= re.sub('Ã©','e', features_limpias)
    features_limpias= re.sub('Ã³','o', features_limpias)
    features_limpias= re.sub('Ãº','u', features_limpias)
    features_limpias= re.sub('Ã±','ñ', features_limpias)#sustituir caracteres por ñ 
    features_limpias= re.sub('\W',' ', features_limpias)#sustituir caracteres especiales por espacios en blanco 
    features_limpias= re.sub('Ã ','i', features_limpias)#sustituir caracteres (vocales con tilde) por vocales
    features_limpias = re.sub(r'\s+', ' ', features_limpias, flags=re.I)#sustituir doble espacio por un espacio 
    features_limpias = features_limpias.lower() #convertir todo a minusculas
    features_limpias2.append(features_limpias)



# In[20]:


len(features_limpias2)


# In[19]:


features_limpias2


# In[13]:


df_amazon['comentarios_limpios']=features_limpias2


# In[16]:


df_amazon.to_pickle('amazon.pickle')


# In[4]:


df_amazon = pd.read_pickle('amazon.pickle')


# In[7]:


df_amazon.head()


# In[6]:


features2 = df_amazon.iloc[:, 3].values
features2


# <a style='color:blue'> CONJUNTO TRAIN Y TEST

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(features2, target, test_size=0.3, random_state=1987)


# In[10]:


print(X_train)
print(y_train)


# <a style='color:blue'> Transformación de texto en números

# In[8]:


#Stopwords limpia el contenido borrando palabras que no son útiles para la predicción
stopwords1 = stopwords.words('spanish')


# In[14]:


print(len(stopwords1))


# In[17]:


#Aquí vemos las palabras que ha borrado para que no estorben en la predicción
stopwords1


# In[9]:


#Transformamos los comentarios en un vector numérico. Entrenamos en train para luego transformar en train y test
vector = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords1)
X_train = vector.fit_transform(X_train).toarray()


# <a style='color:green'> Definimos que el maximo de features que elija para la matriz sea de 2500, lo que significa que suara solo las 2500 palabras mas frecuentes de todos los comentarios para crear la bolsa de palabras del vector. Las palabras que son menos frecuentes no son tan útiles para la clasificación.
# Max_df especifica que solo usará aquellas palabras que ocurran máximo en el 80% de los comentarios, asi mismo las palabras que aparecen en el 100% de los comentarios suelen ser comunes y tampoco aportan a la clasificación.
# Min_df le indica que debe escoger palabras que al menos ocurran en 7 comentarios.

# In[19]:


len(X_train[1])#2500 features seleccionadas


# In[27]:


X_train[1].max()


# In[10]:


#Transformamos también el conjunto de test
X_test=vector.transform(X_test).toarray()


# In[47]:


vector.get_params


# <a style='color:blue'> ESTANDARIZAMOS: Normalizando los datos

# In[12]:


scaler=StandardScaler()


# In[13]:


scaler.fit(X_train)


# In[14]:


X_train_sc=scaler.transform(X_train)


# In[16]:


X_test_sc=scaler.transform(X_test)


# <a style='color:blue'> MODELOS

# <a style='color:blue'> MODELO 1: RANDOM FOREST

# In[11]:


gs_rfc=GSC(estimator=RFC(),
          param_grid={'n_estimators':[200]},
          cv=3,
          verbose=5)


# In[12]:


gs_rfc.fit(X_train,y_train)


# In[13]:


amazon_rfc=gs_rfc.best_estimator_


# In[73]:


with open("amazon_rfc.model", "wb") as archivo_salida:
    pickle.dump(amazon_rfc, archivo_salida)


# In[74]:


with open("amazon_rfc.model", "rb") as archivo_entrada:
    amazon_rfc = pickle.load(archivo_entrada)


# In[27]:


metricas(model_amazon_rfc,X_test,y_test)


# In[28]:


print(classification_report(y_pred=model_amazon_rfc.predict(X_test),y_true =y_test))


# In[29]:


model_amazon_rfc.score(X_train,y_train)


# <a style= 'color:green'>Se ha creado un Grid search para hacer conjuntos de validación y reducir el sobreajuste, aunque no se ha probado con distintos hiperparámetros porque da un fallo de memoria.
# Vemos aun asi que el modelo sobreajusta bastante dando un 99% en train y un 79% en test

# <a style='color:blue'> MODELO 2: LINEALES CON DESCENSO POR GRADIENTE ESTOCÁSTICO

# In[12]:


gs_sgdc=GSC(estimator=SGDC(),
          param_grid={'penalty':['l1','l2']},
          cv=3,
          verbose=5)


# <a style='color:green'> Este modelo implementa modelos de regresión lineal estimando con descenso de gradiente estocastico, la perdida del gradiente se calcula en cada batch. El modelo que lo entrena es una support vector machine.

# In[51]:


gs_sgdc.fit(X_train, y_train)


# In[52]:


amazon_sgdc=gs_sgdc.best_estimator_


# In[53]:


amazon_sgdc


# In[20]:


amazon_sgdc=SGDC()


# In[71]:


with open("amazon_sgdc.model", "wb") as archivo_salida:
    pickle.dump(amazon_sgdc, archivo_salida)


# In[72]:


with open("amazon_sgdc.model", "rb") as archivo_entrada:
    amazon_sgdc = pickle.load(archivo_entrada)


# In[56]:


metricas(model_amazon_sgdc,X_test,y_test)


# In[57]:


model_amazon_sgdc.score(X_train,y_train)


# In[58]:


print(classification_report(y_pred=model_amazon_sgdc.predict(X_test),y_true =y_test))


# <a style= 'color:green'> Este modelo mejora los resultados del Random Forest, con un 80% de accuracy, además no está sobreajustando, (80,2% en train, 79,97% en test),  lo cual nos garantiza que va a generalizar mejor a la hora de predecir.
# 
# Además el tiempo de cómputo de este modelo es mucho más rapido que el del Random Forest

# <a style='color:blue'> MODELO 3: LINEAR con SGD con datos Estandarizados

# <a style='color:green'> Según la teoría para mejores resultados se debe trabajar con datos normalizados, a continuación probaremos el modelo normalizando los datos, y ya que el tiempo de cómputo no nos satura la memoria haremos un grid con más modelos en la función de pérdida. Se probarán: hinge= SVM, squearde_hinge= SVM cuadráticas, función del perceptron, SVM con squared epsilon

# In[27]:


gs_sgdc3=GSC(estimator=SGDC(),
          param_grid={'penalty':['l1','l2'],
                     'loss':['hinge','squared_hinge','perceptron','squared_epsilon_insensitive']},
          cv=5,
          verbose=5)


# In[28]:


gs_sgdc3.fit(X_train_sc,y_train)


# In[29]:


amazon_sgdc3=gs_sgdc3.best_estimator_


# In[30]:


amazon_sgdc3


# <a style='color:green'> La función de pérdida que mejor funciona es SVM

# In[31]:


metricas(amazon_sgdc3,X_test_sc,y_test)


# <a style='color:green'> Pero vemos que con datos estandarizados bajan las métricas del modelo

# <a style='color:blue'> MODELO 4: MLP CLASSIFIER

# In[30]:


gs_mlp=GSC(estimator=MLP(),
          param_grid={'activation':['relu'],
                     'alpha':[0.0001,0.001]},
          cv=3,
          verbose=5)


# In[31]:


gs_mlp.fit(X_train,y_train)


# In[32]:


amazon_mlp=gs_mlp.best_estimator_


# In[68]:


with open("amazon_mlp.model", "wb") as archivo_salida:
    pickle.dump(amazon_mlp, archivo_salida)


# In[69]:


with open("amazon_mlp.model", "rb") as archivo_entrada:
    amazon_mlp = pickle.load(archivo_entrada)


# In[37]:


model_amazon_mlp


# In[36]:


metricas(model_amazon_mlp,X_test,y_test)


# In[49]:


print(classification_report(y_pred=model_amazon_mlp.predict(X_test),y_true =y_test))


# In[59]:


model_amazon_mlp.score(X_train,y_train)


# <a style='color:green'>La accuracy de la red neuronal en train es del 99% mientras que en test del 75%. El modelo está sobreajustando. 

# <a style='color:blue'> MODELO 4: NAIVE BAYES

# Naive Bayes classifier for multinomial models
# The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

# In[30]:


gs_mnb2=GSC(estimator=MNB(),
          param_grid={'fit_prior':[True, False]
                     },
          cv=10,
          verbose=5)


# In[31]:


gs_mnb2.fit(X_train,y_train)


# In[39]:


amazon_mnb2=gs_mnb2.best_estimator_


# In[43]:


amazon_mnb2


# In[41]:


with open("amazon_mnb.model", "wb") as archivo_salida:
    pickle.dump(amazon_mnb2, archivo_salida)


# In[42]:


with open("amazon_mnb.model", "rb") as archivo_entrada:
    amazon_mnb = pickle.load(archivo_entrada)


# In[44]:


metricas(amazon_mnb,X_test,y_test)


# In[45]:


amazon_mnb.score(X_train,y_train)


# <a style='color:green'> Est modelo tampoco sobreajusta. Pero sigue siendo mejor el linear con descenso de gradiente estocástico y función de pérdida SVM

# <a style='color:blue'> MODELO 4: LINEAR SUPPORT VECTOR MACHINE

# <a style='color:green'> Las SVM también funcionan muy bien para clasificar textos

# In[18]:


gs_svc=GSC(estimator=SVC(),
          param_grid={'loss':['hinge','squared_hinge'],
                     'penalty':['l2']},
          cv=3,
          verbose=5)


# In[59]:


gs_svc.fit(X_train,y_train)


# In[60]:


amazon_svm=gs_svc.best_estimator_


# In[61]:


with open ('amazon_svm.model','wb') as archivo_salida:
    pickle.dump(amazon_svm,archivo_salida)


# In[32]:


with open ('amazon_svm.model','rb') as archivo_entrada:
    amazon_svm=pickle.load(archivo_entrada)


# In[33]:


metricas(amazon_svm,X_test,y_test)


# In[34]:


amazon_svm.score(X_train,y_train)


# <a style='color:green'>Efectivamente la SVM es el modelo con mejores métricas entre todos los modelos probados, este modelo tampoco sobreajusta (80.78 en train y 80.37 en test). Garantizando que sabe generalizar en las predicciones.

# <a style='color:blue'> CONCLUSIONES

# <a style='color:green'> El modelo más robusto que hemos probado para este caso es es una SVM Lineal. Este modelo ha sido el que mejor accuracy nos ha dado con un 80,37%. 
# 
# Otro modelo recomendado para predicción de texto son las SVM no lineales, pero estos modelos internamente hacen multiplicaciones matriciales que para este caso al tener mas de 700k comentarios con 2500 features, se volvían muy grandes y daban un fallo de memoria.
# 
# Se intentó hacer pipelines para que vectorizen, estandarizen y entrenen los modelos, pero la memoria de computo fallaba.
# 
