# Machine-Learning
Modelos de clasificación y regresión

# Análisis de sentimientos
La SVM es el modelo con mejores métricas entre todos los modelos probados, este modelo tampoco sobreajusta (80.78 en train y 80.37 en test). Garantizando que sabe generalizar en las predicciones.

El modelo más robusto que hemos probado para este caso es es una SVM Lineal. Este modelo ha sido el que mejor accuracy nos ha dado con un 80,37%. 

Otro modelo recomendado para predicción de texto son las SVM no lineales, pero estos modelos internamente hacen multiplicaciones matriciales que para este caso al tener mas de 700k comentarios con 2500 features, se volvían muy grandes y daban un fallo de memoria.
Se intentó hacer pipelines para que vectorizen, estandarizen y entrenen los modelos, pero la memoria de computo fallaba.

# Modelo de Regresión para predecir la calidad del vino
Con este modelo se pretende predecir la calidad del vino valorada numéricamente en una escala del 1 al 10

El modelo obtenido acierta un 57% de las veces en la predicción de la calidad y el 39% se equivoca con +-1 punto. 
Se equivoca con 2 puntos un 3% de las veces y el máximo error es de 3 puntos, equivocandose solo un 0.3% de veces.
No es un modelo totalmente preciso, como ya se ha comentado las variables no son las mejores para predecir la calidad, pero se puede concluir que 96 de cada 100 veces, acierta un 60% y un 40% se equivoca máximo con + - 1 punto en predecir la calidad. Es decir que es un modelo con un 96% de aciertos con tolerancia de +1-1 punto en la predicción.

# Modelo de Clasificación para predecir el tipo de vino (blanco o rojo)

Con este modelo se pretende predecir el color del vino

Modelo ganador: RANDOM FOREST CLASSIFIER 

Todos los modelos dan métricas realmente buenas. RFC es más robusto por muy poca diferencia. Obtenemos métricas realmente buenas:

Accuracy : 99.37   
Recall  : 99.83       
Precision : 99.34      

