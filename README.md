# Machine-Learning
Machine learning models applied to different problems


# Análisis de sentimientos
La SVM es el modelo con mejores métricas entre todos los modelos probados, este modelo tampoco sobreajusta (80.78 en train y 80.37 en test). Garantizando que sabe generalizar en las predicciones.

El modelo más robusto que hemos probado para este caso es es una SVM Lineal. Este modelo ha sido el que mejor accuracy nos ha dado con un 80,37%. 

Otro modelo recomendado para predicción de texto son las SVM no lineales, pero estos modelos internamente hacen multiplicaciones matriciales que para este caso al tener mas de 700k comentarios con 2500 features, se volvían muy grandes y daban un fallo de memoria.
Se intentó hacer pipelines para que vectorizen, estandarizen y entrenen los modelos, pero la memoria de computo fallaba.

