# Actividad_5-11_ACO


1) Resumen — Arquitecturas Observadas

Arquitectura original del tutorial: red feed-forward de 3 capas con configuración [2, 3, 4] — es decir 2 neuronas de entrada, 1 capa oculta con 3 neuronas, y 4 neuronas de salida (cada salida controla un motor). Las activaciones son tanh (salidas en −1..1 que luego se interpretan como 0/1). 
Aprende Machine Learning

Normalización/Entrada-Salida: las entradas se escalaban a valores en {−1, 0, 1} (ej. posiciones del obstáculo, distancia) compatibles con la tangente hiperbólica; las salidas se interpretaban binariamente (encender/apagar motor). 
Aprende Machine Learning

Copia de pesos a Arduino: el workflow del artículo entrena la red en Python/Jupyter, extrae matrices de pesos y biases y las copia como arrays en el sketch Arduino. En Arduino se hace sólo forward propagation (multiplicaciones+sumas+tanh) para decidir los motores. 
Aprende Machine Learning

2) Enfoques de resolución de problemas aplicados

Aprendizaje supervisado con red pequeña: usar un dataset pequeño (9 ejemplos en el tutorial) y MSE como función de coste para entrenar la red.

Simplificación del problema: discretizar las entradas (−1,0,1) y las salidas (0/1) para hacer el aprendizaje manejable con muy pocos datos.

Transferencia de pesos: entrenar offline (en Python/Colab), exportar pesos a un micro (Arduino) para ejecutar en tiempo real sin capacidad de entrenamiento.

Modularización: separar la parte ML (entrenamiento + exportación) de la parte embedded (sketch Arduino con forward).

Validación por simulación: probar las predicciones de la red con los datos de entrenamiento y con entradas nuevas para comprobar comportamiento antes de conectar motores.

3) Ejecutar el colab para entrenar la red neuronal

Cristian Sasinka: https://colab.research.google.com/drive/1KmYFxev4ETwZrcQcgeLVZUxIJIUz0NrC?usp=sharing

4) Código:
Implementar red neuronal con 3 entradas y 5 salidas para navegación autónoma

Cristian Sasinka: https://colab.research.google.com/drive/144FvNo_sEW4GMvAhU2luXV61SOVEJF3m?usp=sharing

Facundo Castillo: https://colab.research.google.com/drive/1WqVbn1Ufexyrg4WUETd0p7B3bwFFrEfx#scrollTo=RxIKFkDOFrKM
