# Coil-100
Coil100

[SITIO DESPLEGADO_COIL100](https://coil100.lovable.app/)

[Repositorio de sitio web](https://github.com/Fx2048/coil100)




1. Resumen (Abstract)
El presente proyecto aborda la clasificación de objetos mediante técnicas de visión por computadora, comparando un enfoque de visión clásica basado en extracción manual de características y un enfoque de aprendizaje profundo mediante redes neuronales convolucionales, considerando la calidad de predicción y el rendimiento computacional bajo diferentes condiciones de ejecución. Para ello, se utilizó un subconjunto del conjunto de datos COIL-100, compuesto por cinco categorías: taza con sandías (obj043), pimiento rojo (obj063), grapadora (obj068), patito de goma (obj074) y fresa (obj075), con un total de 360 imágenes.
El enfoque clásico combinó los descriptores HOG, LBP e histogramas de color HSV, utilizando Random Forest como clasificador. Por otro lado, el enfoque de aprendizaje profundo empleó MobileNetV2, con pesos preentrenados en ImageNet y una capa clasificadora adaptada a las cinco categorías seleccionadas.
Se establecieron dos escenarios experimentales: uno con ángulos de visualización intercalados y otro con ángulos de prueba no utilizados durante el entrenamiento. Asimismo, se evaluó la extracción de características mediante procesamiento secuencial y paralelo, utilizando dos y cuatro procesos. También se comparó la inferencia de MobileNetV2 en CPU y GPU, se analizaron diferentes tamaños de lote y se evaluó la exportación del modelo al formato ONNX mediante ONNX Runtime.
En el escenario B, Random Forest alcanzó una exactitud del 93,89 %, mientras que MobileNetV2 obtuvo un 100 % con resolución de 128 × 128 píxeles y un 97,78 % con resolución de 64 × 64 píxeles. En cuanto al procesamiento paralelo, se obtuvo una aceleración aproximada de 1,80 veces con dos procesos y de 1,95 veces con cuatro procesos respecto a la ejecución secuencial. Por otro lado, ONNX Runtime alcanzó una aceleración de 6,13 veces frente a PyTorch en CPU.
Finalmente, las pruebas con diferentes tamaños de lote permitieron analizar el rendimiento de inferencia en CPU y GPU. Los resultados muestran que MobileNetV2 obtuvo una mayor calidad de clasificación en las condiciones evaluadas, mientras que ONNX Runtime y el procesamiento paralelo ofrecieron alternativas para mejorar los tiempos de ejecución.



Palabras clave: visión por computadora, clasificación de imágenes, Random Forest, MobileNetV2, HOG, LBP, HSV, procesamiento paralelo, CPU, GPU, ONNX.
2. Introducción
La visión por computadora es un campo que busca desarrollar sistemas capaces de interpretar información visual y reconocer objetos a partir de imágenes digitales. Según Sucar y Gómez (2004), este proceso comprende el análisis de las imágenes para obtener información relevante del entorno y permitir su interpretación automática. En este contexto, el reconocimiento de objetos constituye una aplicación importante de la visión computacional.
Tradicionalmente, este problema se ha abordado mediante técnicas de visión clásica, en las que se extraen características manualmente, como bordes, formas, texturas y colores. Posteriormente, estas características se utilizan como entrada de algoritmos de aprendizaje automático. Entre los descriptores utilizados se encuentran Histogram of Oriented Gradients (HOG), que representa información relacionada con la orientación de los gradientes; Local Binary Patterns (LBP), que permite describir patrones locales de textura; y los histogramas del espacio de color HSV, que representan la distribución de las características cromáticas de las imágenes  (Dalal & Triggs, 2005; Ojala et al., 2002; Sural et al., 2002).
Por otro lado, las redes neuronales convolucionales (CNN) permiten aprender automáticamente representaciones visuales a partir de los datos. Una de estas arquitecturas es MobileNetV2, diseñada para ofrecer un equilibrio entre capacidad de representación y eficiencia computacional (Sandler et al., 2018).
En este proyecto se seleccionó un subconjunto de cinco objetos individuales del conjunto de datos COIL-100: obj043 (taza con sandías), obj063 (pimiento rojo), obj068 (grapadora), obj074 (patito de goma) y obj075 (fresa). Es importante precisar que el objetivo del experimento no consiste en reconocer de manera general cualquier taza, pimiento, grapadora, patito de goma o fresa, sino en clasificar los cinco objetos específicos seleccionados del conjunto COIL-100 a partir de sus imágenes capturadas desde diferentes ángulos de visualización. Por tanto, los resultados obtenidos se limitan a estos objetos y a las condiciones experimentales establecidas.
El principal desafío consiste en analizar cómo se comportan ambos métodos cuando las imágenes de prueba presentan orientaciones que no estuvieron disponibles durante el entrenamiento. Además, se estudia el efecto de distribuir la extracción de características entre varios procesos y se compara el rendimiento de la inferencia de la red neuronal en CPU y GPU.
La comparación no se limita a determinar qué modelo obtiene mayor exactitud, sino que también considera el tiempo de ejecución, la latencia, el throughput y el consumo de memoria. De esta manera, se busca identificar las ventajas y limitaciones de cada alternativa bajo las condiciones experimentales establecidas.

2.1. Objetivo general
Comparar el rendimiento de un método de visión clásica y una red neuronal convolucional para la clasificación de objetos, evaluando su calidad de predicción y comportamiento computacional bajo diferentes condiciones de visualización y ejecución secuencial, paralela y acelerada por GPU.
2.2. Objetivos específicos
Implementar un método de clasificación basado en extracción manual de características HOG, LBP y HSV, utilizando Random Forest.
Implementar una solución de aprendizaje profundo mediante MobileNetV2, adaptada a cinco categorías del conjunto COIL-100.
Evaluar el comportamiento de los modelos ante ángulos de visualización intercalados y ángulos no vistos durante el entrenamiento.
Comparar la extracción secuencial y paralela de características mediante diferentes cantidades de procesos.
Analizar el rendimiento de MobileNetV2 durante la inferencia en CPU y GPU.
Evaluar el efecto de reducir la resolución de entrada de 128 × 128 a 64 × 64 píxeles.
Comparar los resultados mediante métricas de clasificación y rendimiento computacional.
Evaluar el efecto de diferentes tamaños de lote (batch size) sobre el rendimiento de inferencia de MobileNetV2 en CPU y GPU, considerando la latencia y la cantidad de imágenes procesadas por segundo.
Exportar el modelo MobileNetV2 al formato ONNX y comparar su rendimiento de inferencia con PyTorch en CPU, verificando la equivalencia de las predicciones.
2.3. Pregunta de investigación
¿En qué condiciones un pipeline clásico ofrece una relación precisión–costo más favorable que una CNN, y cuándo la red neuronal se vuelve claramente superior? 
3. Antecedentes y trabajos relacionados
3.1. Visión clásica por computadora
La visión clásica utiliza técnicas de procesamiento de imágenes para obtener características que permitan representar los objetos. Estas características pueden estar relacionadas con el color, la textura, los bordes y la forma.
En este proyecto se combinaron tres tipos de descriptores: HOG, LBP e histogramas HSV. La finalidad de esta combinación fue representar diferentes aspectos visuales de las imágenes y proporcionar información al clasificador Random Forest.
Una limitación de este enfoque es que su rendimiento depende de la capacidad de los descriptores seleccionados para representar adecuadamente los objetos ante cambios de orientación y otras variaciones visuales.
3.2. Histogram of Oriented Gradients (HOG)
HOG es un descriptor que representa la distribución de las orientaciones de los gradientes de una imagen. Permite describir características relacionadas con la estructura y los contornos de los objetos.
Dalal y Triggs (2005) estudiaron la utilización de histogramas de gradientes orientados para la detección de personas, mostrando la utilidad de esta representación para describir estructuras visuales.
En el proyecto se configuró HOG con nueve orientaciones, celdas de 16 × 16 píxeles y bloques de 2 × 2 celdas.
3.3. Local Binary Patterns (LBP)
LBP es un descriptor de textura que utiliza comparaciones entre los valores de intensidad de un píxel y los de sus vecinos. Estas comparaciones permiten representar patrones locales de la imagen.
Ojala et al. (2002) desarrollaron métodos basados en patrones binarios locales para el análisis de texturas.
En el proyecto se utilizó LBP con ocho vecinos, radio de un píxel y el método uniforme (uniform).
3.4. Histogramas de color HSV
El espacio de color HSV representa las imágenes mediante tres componentes:
Hue: tono o tipo de color.
Saturation: intensidad o saturación del color.
Value: valor o brillo.
Los histogramas permiten representar la distribución de los colores presentes en una imagen. En este proyecto se utilizaron 32 intervalos para cada componente, generando un total de 96 características de color.
3.5. Random Forest
Random Forest es un algoritmo de aprendizaje automático que utiliza múltiples árboles de decisión para realizar predicciones. Su resultado se obtiene combinando las predicciones de los árboles que conforman el modelo.
Breiman (2001) presentó este método como una técnica de clasificación y regresión basada en conjuntos de árboles aleatorios.
En el proyecto se utilizó Random Forest con 100 árboles, una semilla aleatoria de 42 y n_jobs=1 para mantener secuencial el entrenamiento del clasificador.
3.6. Redes neuronales convolucionales y MobileNetV2
Las redes neuronales convolucionales permiten aprender características visuales mediante capas que procesan información espacial de las imágenes.
MobileNetV2 es una arquitectura diseñada para reducir el costo computacional de las redes convolucionales, manteniendo una capacidad de representación útil para tareas de visión por computadora (Sandler et al., 2018).
En el proyecto se utilizó MobileNetV2 con pesos preentrenados en ImageNet. La capa clasificadora original fue reemplazada por una nueva capa con cinco salidas, correspondientes a las categorías seleccionadas.
Se utilizó el enfoque de transferencia de aprendizaje, aprovechando los pesos preentrenados como punto de partida y ajustando los parámetros del modelo durante el entrenamiento.
3.7. Relación de los antecedentes con la propuesta
Los métodos clásicos y las redes convolucionales representan dos alternativas para abordar la clasificación de imágenes. El primero utiliza descriptores definidos manualmente, mientras que el segundo aprende representaciones visuales durante el entrenamiento.
La propuesta combina ambos enfoques para estudiar sus diferencias en un mismo problema de clasificación. Además, incorpora procesamiento paralelo en la extracción de características y una comparación de inferencia en CPU y GPU, con el propósito de analizar no solo la calidad de las predicciones, sino también el comportamiento computacional
4. Metodología y diseño de la solución
4.1. Conjunto de datos
Se utilizó el conjunto de datos COIL-100 (Columbia Object Image Library), que contiene imágenes de 100 objetos diferentes capturados desde distintos ángulos de visualización (Nene et al., 1996).
Aunque el conjunto original contiene 7200 imágenes, para el desarrollo experimental se seleccionaron cinco categorías:
Tabla 1. Categorías utilizadas en el experimento
Código
Categoría
obj043
Taza con sandías
obj063
Pimiento rojo
obj068
Grapadora
obj074
Patito de goma
obj075
Fresa


Cada categoría contiene 72 imágenes, por lo que el subconjunto experimental está compuesto por 360 imágenes.
Las imágenes fueron organizadas en carpetas según su categoría y conjunto de entrenamiento o prueba. La resolución principal utilizada fue de 128 × 128 píxeles. Adicionalmente, se realizó una evaluación de MobileNetV2 con una resolución de 64 × 64 píxeles.

4.2. Preparación de los escenarios experimentales
Se establecieron dos escenarios para analizar el comportamiento de los modelos ante diferentes condiciones de orientación de los objetos.
4.2.1. Escenario A: ángulos intercalados
En este escenario, las imágenes se distribuyeron entre entrenamiento y prueba mediante bloques intercalados de ángulos de visualización.
La distribución fue la siguiente:
Entrenamiento: 60 imágenes por categoría.
Prueba: 12 imágenes por categoría.
Total de entrenamiento: 300 imágenes.
Total de prueba: 60 imágenes.
Este escenario representa una condición en la que el modelo dispone de diferentes orientaciones durante el entrenamiento.
En este escenario se evaluó el método clásico mediante Random Forest.
4.2.2 Escenario B: ángulos no vistos
En este escenario, las imágenes se dividieron según su ángulo de visualización:
Entrenamiento: imágenes con ángulos de 0° a 175°.
Prueba: imágenes con ángulos de 180° a 355°.
La distribución fue:
Entrenamiento: 36 imágenes por categoría.
Prueba: 36 imágenes por categoría.
Total de entrenamiento: 180 imágenes.
Total de prueba: 180 imágenes.
Este escenario permite analizar el comportamiento de los modelos ante orientaciones que no estuvieron presentes durante el entrenamiento.
Tabla 2. Distribución de imágenes por escenario

Escenario 
Entrenamiento 
Prueba 
Total 
A: ángulos intercalados 
300 
60
360
B: ángulos no vistos 
180
180
360


4.3. Arquitectura general de la solución
La solución se organizó en diferentes etapas: selección y preparación de imágenes, extracción de características, clasificación y evaluación de resultados.
Se implementaron dos ramas principales:
Rama clásica: extracción de características HOG, LBP y HSV, seguida del entrenamiento y evaluación de Random Forest.
Rama de aprendizaje profundo: preparación de imágenes, entrenamiento de MobileNetV2 y evaluación de la inferencia en CPU y GPU.
Adicionalmente, se incorporó una etapa de procesamiento paralelo para distribuir la extracción de características del método clásico entre varios procesos.
Figura 1: Esquema metodológico para la comparación de enfoques de visión clásica y aprendizaje profundo.

4.4. Método de visión clásica
El método clásico se desarrolló mediante la extracción manual de características visuales y un clasificador Random Forest.
Cada imagen fue redimensionada a 128 × 128 píxeles. Posteriormente, se calcularon tres grupos de características:

a) HOG
Se utilizó para representar la estructura de los objetos mediante la distribución de las orientaciones de los gradientes.
Configuración:
Orientaciones: 9.
Tamaño de celda: 16 × 16 píxeles.
Bloques: 2 × 2 celdas.
b) LBP
Se utilizó para representar patrones locales de textura.
Configuración:
Número de vecinos: 8.
Radio: 1 píxel.
Método: uniforme (uniform).
c) Histogramas HSV
Se utilizaron para representar la distribución del color.
Configuración:
Hue: 32 intervalos.
Saturation: 32 intervalos.
Value: 32 intervalos.
Total: 96 características de color.
La combinación de los tres descriptores generó un vector de 1870 características por imagen.
Posteriormente, este vector fue utilizado como entrada del clasificador Random Forest.
Tabla 3: Configuración del método clásico 

Elemento 
Configuración 
Resolución 
128 × 128 píxeles 
Descriptor HOG 
9 orientaciones 
Celdas HOG 
16 × 16 píxeles 
Bloques HOG 
2 × 2 celdas 
Descriptor LBP 
8 vecinos, radio 1 
Método LBP 
Uniforme 
Histograma HSV 
32 intervalos por componente 
Características totales 
1870 
Clasificador 
Random Forest 
Número de árboles 
100
Semilla aleatoria 
42
Entrenamiento 
Secuencial, n_jobs=1 


4.5. Método basado en redes neuronales convolucionales
El segundo enfoque utilizó MobileNetV2, una arquitectura convolucional preentrenada en ImageNet.
Se reemplazó la capa clasificadora original por una nueva capa con cinco salidas, correspondientes a las categorías seleccionadas del conjunto COIL-100.
La configuración utilizada fue:






Tabla 4. Configuración de MobileNetV2
Elemento 
Configuración 
Modelo CNN 
MobileNetV2 
Total de imágenes 
360
Pesos iniciales 
Preentrenados en ImageNet 
Resolución principal 
128 × 128 píxeles 
Resolución adicional 
64 × 64 píxeles 
Tamaño del lote 
32
Épocas 
8
Optimizador 
Adam 
Tasa de aprendizaje 
0.0001 
Función de pérdida 
Cross Entropy Loss 
Número de parámetros 
2 230 277 
Tamaño aproximado de parámetros 
8,5 MB 


El entrenamiento se realizó utilizando PyTorch y GPU. Durante el proceso, los parámetros del modelo fueron optimizados mediante el algoritmo Adam.
Posteriormente, el modelo fue evaluado en CPU y GPU utilizando las imágenes de prueba del escenario B.
También se entrenó una variante con resolución de 64 × 64 píxeles para analizar el efecto de reducir el tamaño de entrada.
4.6. Procesamiento secuencial y paralelo
Se evaluaron tres configuraciones de procesamiento para la extracción de características:
Procesamiento secuencial: ejecución de las tareas mediante un único proceso, utilizado como referencia de comparación.
Procesamiento paralelo con dos procesos: distribución del trabajo entre dos procesos independientes.
Procesamiento paralelo con cuatro procesos: distribución del trabajo entre cuatro procesos independientes.
Esta comparación permitió analizar el efecto de la paralelización sobre el tiempo de extracción de características, manteniendo el mismo procedimiento de cálculo de los descriptores.
4.7. Comprobación de equivalencia
Antes de comparar los tiempos de ejecución, se verificó que la extracción secuencial y la extracción paralela produjeran resultados equivalentes.
Para ello, se utilizó la función numpy.allclose() sobre los vectores de características obtenidos mediante ambas modalidades.
En las configuraciones evaluadas, el resultado de la comprobación fue verdadero (True), indicando que los resultados de extracción fueron equivalentes dentro de la tolerancia numérica utilizada.
Esta comprobación permitió comparar el rendimiento de las dos modalidades sin atribuir a la paralelización cambios en las características generadas.
5. Implementación
La implementación del proyecto contempla el desarrollo de dos enfoques de clasificación de imágenes: visión clásica mediante Random Forest y aprendizaje profundo mediante MobileNetV2. Ambos métodos se aplicaron sobre cinco objetos seleccionados del conjunto de datos COIL-100. Asimismo, se implementaron diferentes pruebas experimentales para analizar la calidad de clasificación y el rendimiento computacional, considerando distintos escenarios de partición de imágenes, procesamiento secuencial y paralelo, ejecución en CPU y GPU, tamaños de lote y exportación del modelo a formato ONNX. Finalmente, se desarrolló un dashboard web que permite interactuar con el modelo y realizar pruebas de clasificación.
5.1. Implementación secuencial
La implementación secuencial constituye la referencia para evaluar el rendimiento del método clásico.
El procedimiento comienza con la lectura de las imágenes y su redimensionamiento a 128 × 128 píxeles. Posteriormente, se calculan las características HOG, LBP y HSV y se combinan en un único vector de 1870 elementos.
Una vez obtenidas las características del conjunto de entrenamiento, se entrena el clasificador Random Forest con 100 árboles y n_jobs=1.
Finalmente, se extraen las características de cada imagen de prueba y se realiza la predicción de su categoría.
5.2. Implementación paralela
La implementación paralela se aplicó a la extracción de características de las imágenes del método clásico.
Se utilizó multiprocessing.Pool para distribuir las rutas de las imágenes entre los procesos disponibles. Cada proceso ejecuta la función de extracción de características de forma independiente.
La distribución se realizó mediante pool.map(), utilizando un tamaño de bloque (chunksize) de 8.
Las configuraciones evaluadas fueron:
Un proceso: referencia secuencial.
Dos procesos: distribución del trabajo entre dos procesos.
Cuatro procesos: distribución del trabajo entre cuatro procesos.
La implementación paralela no modificó el algoritmo de extracción de características ni los descriptores utilizados. Su propósito fue reducir el tiempo necesario para procesar las imágenes.
El entrenamiento de Random Forest y de MobileNetV2 no fue distribuido entre varios procesos mediante esta implementación. Por tanto, la evaluación del paralelismo corresponde específicamente a la etapa de extracción de características del método clásico.
5.3. Comparación CPU y GPU
Para evaluar el comportamiento de MobileNetV2 se realizaron pruebas de inferencia en CPU y GPU.
El modelo fue entrenado utilizando CUDA, según el registro de ejecución del notebook. Posteriormente, se evaluó en CPU y GPU utilizando las mismas 180 imágenes de prueba del escenario B.
La evaluación incluyó la carga de cada imagen, su transformación y la ejecución del modelo para obtener la categoría predicha.
En el caso de GPU, se realizaron cinco inferencias de calentamiento antes de medir las latencias. Además, se utilizó la sincronización implícita mediante .item() para obtener la predicción y registrar el tiempo correspondiente.
No se midió de forma independiente el tiempo de transferencia de datos entre CPU y GPU. Por ello, los tiempos reportados corresponden a la medición de la operación de evaluación definida en el notebook y no deben interpretarse como tiempo puro de cómputo del modelo.
5.4. Entorno de ejecución y configuración experimental
Para desarrollar y ejecutar los experimentos se utilizó la plataforma Kaggle, bajo un entorno basado en Linux con arquitectura x86_64. El entorno computacional contó con un procesador de 2 núcleos físicos y 4 hilos lógicos, además de 31,35 GB de memoria RAM.
En cuanto a los recursos de procesamiento gráfico, se dispuso de dos GPU NVIDIA Tesla T4, con soporte para CUDA 13.0. Estos recursos permitieron ejecutar los experimentos de aprendizaje profundo y realizar las comparaciones de rendimiento entre CPU y GPU.
Las principales bibliotecas utilizadas fueron PyTorch 2.11.0+cu130, Torchvision 0.26.0+cu130, scikit-learn 1.6.1, OpenCV 4.14.0 y NumPy 2.1.3. Asimismo, se utilizó Python 3.13.16 como lenguaje de programación. Estas herramientas permitieron implementar los modelos de clasificación, realizar el procesamiento de imágenes, extraer características y evaluar los resultados obtenidos.
La configuración experimental de los modelos consideró ocho épocas de entrenamiento, un tamaño de lote (batch size) de 32 imágenes y una tasa de aprendizaje de 0.0001 para MobileNetV2. Para Random Forest se utilizaron 100 árboles de decisión y una semilla aleatoria de 42, con el propósito de favorecer la reproducibilidad de los experimentos.
Finalmente, se realizaron evaluaciones de clasificación, extracción de características y medición del tiempo de inferencia en CPU y GPU. Asimismo, se contempló la exportación del modelo a formato ONNX para evaluar su compatibilidad con otros entornos de ejecución, siempre que dicha operación se haya realizado efectivamente.
5.5. Exportación y validación del modelo ONNX
Con la finalidad de evaluar una alternativa de ejecución del modelo de aprendizaje profundo, se realizó la exportación de MobileNetV2, con una resolución de entrada de 128 × 128 píxeles, al formato ONNX (Open Neural Network Exchange), que permite la interoperabilidad entre diferentes entornos de ejecución.
Posteriormente, el modelo exportado fue ejecutado mediante ONNX Runtime, utilizando el proveedor de ejecución CPU (CPUExecutionProvider). Se verificó su estructura y se compararon las salidas obtenidas con las del modelo original implementado en PyTorch.
Para evaluar la equivalencia entre ambos entornos, se utilizaron 50 imágenes del conjunto de prueba. Se analizaron la coincidencia de las predicciones y las diferencias numéricas entre las salidas generadas por cada modelo.
Finalmente, se realizó un benchmark de cinco repeticiones para comparar el rendimiento de inferencia de ONNX Runtime y PyTorch en CPU, considerando dos métricas: latencia y throughput.


5.6. Preparación del paquete de resultados para la plataforma web
Como parte de la implementación, se preparó un paquete de archivos para facilitar la utilización y presentación de los resultados en una plataforma web. El paquete incluye los modelos entrenados, los archivos de resultados experimentales, las figuras generadas durante la evaluación y muestras de imágenes correspondientes a las cinco categorías seleccionadas.
Finalmente, los archivos se organizaron en una estructura de directorios y se comprimieron en un archivo ZIP denominado (coil100_lab_export_1.zip), facilitando su distribución y posterior utilización.
6. Análisis de resultados
6.1. Métricas de evaluación
Para analizar la calidad de clasificación se utilizaron las siguientes métricas:
Accuracy: representa la proporción de predicciones correctas respecto al total de imágenes evaluadas.
Precisión macro: calcula la precisión de cada categoría y obtiene el promedio entre ellas.
Recall macro: mide la capacidad del modelo para identificar correctamente las imágenes pertenecientes a cada categoría.
F1 macro: representa el promedio de la medida armónica entre precisión y recall de las categorías.
Para analizar el rendimiento computacional se utilizaron:
Latencia media: tiempo promedio necesario para procesar una imagen.
Latencia P95: valor por debajo del cual se encuentra el 95 % de las mediciones de latencia.
Throughput: cantidad de imágenes procesadas por segundo.
Memoria RAM del proceso: memoria residente registrada durante la ejecución.
Memoria GPU asignada: memoria de GPU asignada por PyTorch durante la evaluación, cuando estuvo disponible.
Para la comparación secuencial y paralela se utilizaron:
S(p)= Ts/Tp,   E(p) = S(p)/p,
Donde:
S(p): Aceleración
E(p): Eficiencia(p)
Ts: Tiempo secuencial
Tp: Tiempo de ejecución paralelo con (p) procesos.
(p): Número de procesos utilizados.

6.2. Resultados de clasificación
6.2.1. Resultados del escenario A
En el escenario A se evaluó Random Forest utilizando ángulos intercalados entre entrenamiento y prueba.
El modelo fue entrenado con 300 imágenes y evaluado con 60 imágenes, distribuidas de manera equilibrada entre las cinco categorías.
Los resultados fueron:
Tabla 5. Tabla de resultados de Random Forest
Métrica 
Resultado 
Accuracy 
100%
Precisión macro 
100%
Recall macro 
100%
F1 macro 
100%
Imágenes de prueba 
60


El modelo clasificó correctamente las 60 imágenes de prueba.
Sin embargo, este resultado corresponde a un escenario en el que el entrenamiento contiene imágenes de diferentes orientaciones. Por ello, no debe interpretarse como evidencia suficiente de generalización ante ángulos completamente no vistos.
Figura 2. Matriz de confusión de Random Forest en el escenario A.
La matriz de confusión del modelo Random Forest en el escenario A evidencia una clasificación correcta de las cinco clases de objetos evaluadas. Se obtuvieron 12 predicciones correctas para cada clase (obj043, obj063, obj068, obj074 y obj075), sin registrar errores de clasificación. Los resultados se concentran completamente en la diagonal principal, lo que demuestra que el modelo identificó correctamente todas las imágenes del conjunto de prueba en este escenario. 
6.2.2. Resultados del escenario B
En el escenario B se evaluaron Random Forest y MobileNetV2 utilizando 180 imágenes de entrenamiento y 180 imágenes de prueba.





Tabla 6. Comparación de calidad de clasificación en el escenario B 
Modelo 
Resolución 
Accuracy 
Precisión macro 
Recall macro 
F1 macro 
Random Forest 
128 × 128 
93,89 % 
95,32 % 
93,89 % 
93,74 % 
MobileNetV2 CPU 
128 × 128 
100 % 
100 % 
100 % 
100 % 
MobileNetV2 GPU 
128 × 128 
100 % 
100 % 
100 % 
100 % 
MobileNetV2 CPU 
64 × 64 
97,78 % 
99,46 % 
99,44 % 
99,44 % 


Los resultados muestran que Random Forest obtuvo una exactitud de 93,89 % en el escenario B, mientras que MobileNetV2 alcanzó el 100 % con resolución de 128 × 128 píxeles, tanto en CPU como en GPU.
La variante de MobileNetV2 con resolución de 64 × 64 píxeles obtuvo una exactitud de 97,78 %, con una pequeña disminución respecto a la configuración de 128 × 128 píxeles.
Estos resultados indican que, bajo las condiciones evaluadas, MobileNetV2 obtuvo una mayor calidad de clasificación que el método clásico en el escenario de ángulos no vistos.
Figura 3. Matriz de confusión de Random Forest en el escenario B.
La matriz de confusión del modelo Random Forest en el escenario B muestra un desempeño favorable en la clasificación de los objetos evaluados. Las clases obj043, obj063, obj068 y obj074 alcanzaron 36 predicciones correctas cada una. Sin embargo, en la clase obj075 se identificaron 25 imágenes correctamente y 11 fueron clasificadas erróneamente como obj063. Este comportamiento evidencia una dificultad del modelo para diferenciar determinados objetos bajo las condiciones de evaluación del escenario B. 
6.2.3. Análisis por categoría del método clásico
En el escenario B, Random Forest presentó resultados diferentes según la categoría.




Tabla 7: Métricas de clasificación por categoría del modelo Random Forest en el escenario B. 
Categoría 
Precisión 
Recall 
F1 
obj043: Taza con sandías 
100%
100%
100%
obj063: Pimiento rojo 
76,60%
100%
86,75%
obj068: Grapadora 
100%
100%
100%
obj074: Patito de goma 
100%
100%
100%
obj075: Fresa 
100%
69,44%
81,97%


Los resultados evidencian que las principales dificultades del método clásico se presentaron en las categorías pimiento rojo y fresa.
En el caso del pimiento rojo, el recall fue del 100 %, pero la precisión fue del 76,60 %. Esto significa que todas las imágenes reales de esa categoría fueron identificadas, aunque algunas imágenes pertenecientes a otras categorías fueron clasificadas incorrectamente como pimiento rojo.
En el caso de la fresa, la precisión fue del 100 %, pero el recall fue del 69,44 %. Esto indica que algunas imágenes reales de fresa fueron clasificadas como otras categorías.
Estos errores muestran que la representación manual de características tuvo dificultades para diferenciar determinadas imágenes cuando se evaluaron orientaciones no utilizadas durante el entrenamiento.
6.2.4. Resultados del método basado en CNN (MobileNetV2)
Para evaluar el desempeño del método basado en aprendizaje profundo, se implementó la arquitectura MobileNetV2 mediante aprendizaje por transferencia (transfer learning), utilizando pesos preentrenados en ImageNet. Se modificó la capa clasificadora final para adaptarla a las cinco categorías del conjunto de datos COIL-100: taza con sandías, pimiento rojo, grapadora, patito de goma y fresa.
El modelo fue configurado con el optimizador Adam, una tasa de aprendizaje de 0.0001, una función de pérdida CrossEntropyLoss y un tamaño de lote (batch size) de 32. El entrenamiento se realizó durante ocho épocas utilizando GPU. Posteriormente, se evaluó el modelo en el escenario B, considerando dos resoluciones de entrada: 128 × 128 y 64 × 64 píxeles.
6.2.4.1. Resultados con resolución de 128 × 128 píxeles
En esta configuración, MobileNetV2 alcanzó una exactitud del 100 %, una precisión macro del 100 %, un recall macro del 100 % y una puntuación F1 macro del 100 %, tanto en la evaluación realizada en CPU como en GPU. Estos resultados indican que el modelo clasificó correctamente las 180 imágenes del conjunto de prueba del escenario B en ambas configuraciones de ejecución, sin registrar errores de clasificación entre las cinco clases evaluadas.



Figura 4: Matrices de confusión de MobileNetV2 (128 × 128) con ejecución en CPU y GPU 
Las matrices de confusión de MobileNetV2, utilizando una resolución de entrada de 128 × 128 píxeles y ejecutadas en CPU y GPU, presentan resultados idénticos en el escenario B. En ambas modalidades, las cinco clases evaluadas registraron 36 predicciones correctas cada una, sin observarse errores ni confusiones entre categorías. Estos resultados evidencian que el modelo logró clasificar correctamente todos los objetos del conjunto de prueba, manteniendo el mismo desempeño predictivo independientemente del dispositivo de ejecución. 
6.2.4.2. Resultados con resolución de 64 × 64 píxeles
Posteriormente, se evaluó MobileNetV2 utilizando imágenes de entrada de 64 × 64 píxeles. En esta configuración, el modelo alcanzó una exactitud del 97,78 % en el escenario B.
Este resultado evidencia una disminución en el rendimiento de clasificación respecto a la configuración de 128 × 128 píxeles, que obtuvo una exactitud del 100 %. La reducción de la resolución de entrada disminuyó la cantidad de información visual procesada por el modelo y favoreció el tiempo de inferencia. En particular, la latencia media en CPU pasó de 16,31 ms a 11,74 ms por imagen. Sin embargo, esta reducción también afectó ligeramente la capacidad de clasificación de determinadas categorías.
De acuerdo con la matriz de confusión y las métricas individuales, se identificaron cuatro imágenes de la clase obj075 clasificadas incorrectamente como obj063. Esta confusión se refleja en un recall del 88,89 % para obj075 y una precisión del 90 % para obj063.
En conjunto, los resultados muestran que MobileNetV2 mantuvo una elevada exactitud con imágenes de menor resolución, aunque presentó una ligera pérdida de rendimiento en la diferenciación de determinadas clases.


Figura 5. Matriz de confusión de MobileNetV2 (64 × 64) – CPU 
La matriz de confusión de MobileNetV2 con resolución de entrada de 64 × 64 píxeles muestra un desempeño elevado en la clasificación de los objetos. Las clases obj043, obj063, obj068 y obj074 registraron 36 predicciones correctas cada una. Por su parte, la clase obj075 obtuvo 32 predicciones correctas y presentó 4 clasificaciones erróneas como obj063. A pesar de esta confusión puntual, el modelo mantuvo una elevada capacidad de reconocimiento, incluso utilizando una resolución de imagen inferior.
6.3. Resultados del rendimiento computacional
Se analizaron los tiempos de entrenamiento, la latencia media, la latencia P95 y el throughput de los modelos evaluados, con la finalidad de comparar su comportamiento computacional bajo diferentes configuraciones de ejecución. Asimismo, se examinó el rendimiento de MobileNetV2 utilizando CPU y GPU, considerando las resoluciones de entrada de 128 × 128 y 64 × 64 píxeles.
Tabla 8. Rendimiento computacional de los modelos evaluados
Configuración 
Tiempo de entrenamiento o ajuste 
Latencia media 
Latencia P95 
Throughput 
Random Forest A 
2,829 s
14,251 ms 
16,430 ms 
70,172 imágenes/s 
Random Forest B
1,694 s 
14,073 ms 
16,568 ms 
71,059 imágenes/s 
MobileNetV2 CPU, 128 × 128 
8,848 s 
16,313 ms 
17,565 ms 
61,301 imágenes/s 
MobileNetV2 GPU, 128 × 128 
8,848 s 
7,119 ms 
7,835 ms 
140,467 imágenes/s 
MobileNetV2 CPU, 64 × 64
2,719 s 
11,736 ms 
12,478 ms 
85,209 imágenes/s 

Nota. Los tiempos de entrenamiento de Random Forest incluyen la extracción de características y el ajuste del clasificador. En MobileNetV2, el tiempo corresponde al entrenamiento del modelo realizado mediante GPU. Las mediciones de latencia y throughput corresponden a los respectivos procedimientos de evaluación implementados en el notebook.
Los resultados muestran diferencias en el rendimiento computacional de los modelos evaluados. Random Forest registró tiempos de entrenamiento total de 2,83 segundos en el escenario A y 1,69 segundos en el escenario B. Por su parte, MobileNetV2 con resolución de 128 × 128 píxeles presentó un tiempo de entrenamiento de 8,85 segundos. Esta diferencia debe interpretarse considerando que ambos enfoques realizan procedimientos de entrenamiento distintos.
En cuanto a la inferencia de MobileNetV2 con resolución de 128 × 128 píxeles, la ejecución en GPU obtuvo una latencia media de 7,12 ms por imagen, mientras que en CPU se registraron 16,31 ms. Asimismo, el throughput aumentó de 61,30 imágenes por segundo en CPU a 140,47 imágenes por segundo en GPU. Estos resultados evidencian un mejor rendimiento de inferencia mediante GPU en las condiciones experimentales evaluadas, manteniendo una exactitud del 100 % en ambos dispositivos.
Por otro lado, la variante de MobileNetV2 con resolución de 64 × 64 píxeles presentó una latencia media de 11,74 ms y un throughput de 85,21 imágenes por segundo en CPU. En comparación con la resolución de 128 × 128 píxeles ejecutada en el mismo dispositivo, esta configuración redujo el tiempo de procesamiento por imagen. Sin embargo, esta mejora debe analizarse conjuntamente con la disminución de la exactitud obtenida por el modelo.
Adicionalmente, se evaluó el efecto del tamaño del lote (batch size) durante la inferencia de MobileNetV2, utilizando tamaños de 1, 8, 16 y 32 imágenes. En CPU, el mayor throughput se obtuvo con un tamaño de lote de 16, alcanzando 124,30 imágenes por segundo. En GPU, el mejor resultado se registró con un tamaño de lote de 32, alcanzando 619,27 imágenes por segundo. Estos resultados muestran que el procesamiento por lotes influye en el rendimiento de inferencia y que la configuración más conveniente depende del dispositivo utilizado.
Es importante señalar que las mediciones de latencia corresponden a los procedimientos de evaluación definidos para cada modelo. En Random Forest se considera la extracción de características y la predicción, mientras que en MobileNetV2 se incluye la carga, transformación y evaluación de las imágenes. Por ello, los valores deben interpretarse teniendo en cuenta las operaciones realizadas en cada enfoque y no como una comparación de tiempos de cómputo estrictamente equivalentes.
6.4. Resultados de memoria
Se evaluó el consumo de memoria durante la ejecución de los modelos, considerando la memoria RAM residente del proceso y, cuando estuvo disponible, la memoria GPU asignada por PyTorch. Esta evaluación permitió identificar las diferencias en el uso de recursos computacionales entre Random Forest y MobileNetV2, así como entre las distintas configuraciones de ejecución.







Tabla 9. Memoria registrada durante la evaluación de los modelos
Configuración 
Memoria RAM antes de la evaluación (MB)
Memoria RAM máxima observada o posterior (MB)
Random Forest A 
867,80
869,54
Random Forest B
870,77
871,14
MobileNetV2 CPU, 128 × 128 
1735,95
1736,91
MobileNetV2 GPU, 128 × 128 
1746,84
1747,22
MobileNetV2 CPU, 64× 64
1785,04
1785,05

Nota. En MobileNetV2, el segundo valor corresponde al máximo de memoria RAM observado mediante muestreo durante la evaluación. En Random Forest, corresponde a la memoria registrada después de la evaluación, debido a que el notebook no proporciona una medición equivalente de máximo observado.
En cuanto al consumo de memoria RAM, los resultados muestran que MobileNetV2 presentó valores superiores a los registrados para Random Forest dentro del entorno experimental. Para la resolución de 128 × 128 píxeles, MobileNetV2 registró una memoria RAM máxima observada de 1736,91 MB en CPU y 1747,22 MB en GPU. Por su parte, la variante de 64 × 64 píxeles alcanzó un valor de 1785,05 MB durante la evaluación. En el caso de Random Forest, se registraron valores de memoria RAM de 869,54 MB y 871,14 MB después de la evaluación de los escenarios A y B, respectivamente. No obstante, estos valores no representan máximos observados mediante muestreo, por lo que su comparación con los registros de MobileNetV2 debe realizarse considerando esta diferencia metodológica. Adicionalmente, durante la ejecución de MobileNetV2 en GPU se registró una memoria máxima asignada por PyTorch de aproximadamente 38,28 MB. Este indicador corresponde a la memoria GPU asignada durante la evaluación y debe analizarse de manera independiente de la memoria RAM del proceso.
En general, los resultados permiten observar diferencias en el uso de recursos de memoria entre los enfoques evaluados. Sin embargo, es importante precisar que la memoria RAM registrada corresponde al proceso completo de ejecución y no exclusivamente al modelo. Por esta razón, estos valores no deben confundirse con el tamaño del archivo del modelo ni con la cantidad de memoria GPU asignada.
6.5. Resultados del procesamiento paralelo
Para evaluar el efecto del procesamiento paralelo se midió el tiempo de extracción de características sobre 360 imágenes. Se realizaron tres repeticiones por configuración y se utilizó la mediana de los tiempos.
Tabla 10. Comparación de extracción secuencial y paralela
Número de procesos 
Tiempo 
Speedup 
Eficiencia 
1
2,574 s 
1,00× 
100%
2
1,427 s 
1,80× 
90,18%
4
1,321 s 
1,95× 
48,69%


Los resultados muestran que el procesamiento paralelo permitió reducir el tiempo de extracción de características.
Con dos procesos, el tiempo disminuyó de 2,574 a 1,427 segundos, alcanzando una aceleración de 1,80 veces y una eficiencia del 90,18 %.
Con cuatro procesos, el tiempo disminuyó a 1,321 segundos, obteniéndose una aceleración de 1,95 veces y una eficiencia del 48,69 %.
Aunque cuatro procesos consiguieron el menor tiempo de ejecución, la eficiencia fue inferior a la obtenida con dos procesos. Esto indica que el incremento del número de procesos no produjo una mejora proporcional al número de trabajadores utilizados.
Además, la comprobación mediante numpy.allclose() confirmó que los resultados de la extracción secuencial y paralela fueron equivalentes en las configuraciones evaluadas.

Figura 6. Aceleración del procesamiento paralelo
6.6. Evaluación de la resolución de entrada
También se evaluó MobileNetV2 con resoluciones de 128 × 128 y 64 × 64 píxeles.
La configuración de 128 × 128 alcanzó una exactitud del 100 % en el escenario B, mientras que la variante de 64 × 64 obtuvo 99,44 %.
La reducción de resolución disminuyó la latencia media de inferencia en CPU de 19,294 ms a 14,089 ms por imagen.
Sin embargo, también se observó una pequeña disminución en la calidad de clasificación. En la variante de 64 × 64, el recall de obj063 fue del 97 %, mientras que la precisión de obj075 fue del 97 %.
Estos resultados muestran que reducir la resolución puede mejorar el tiempo de procesamiento, aunque puede producir pequeñas pérdidas en la calidad de clasificación.



6.7. Síntesis de los resultados
Los experimentos muestran que Random Forest alcanzó una exactitud del 100 % en el escenario A, donde las orientaciones estuvieron intercaladas entre entrenamiento y prueba. Sin embargo, en el escenario B su exactitud disminuyó al 93,89 %.
MobileNetV2 obtuvo una exactitud del 100 % en el escenario B con resolución de 128 × 128 píxeles, tanto en CPU como en GPU. La variante de 64 × 64 alcanzó una exactitud del 99,44 %.
En términos de rendimiento, la inferencia de MobileNetV2 en GPU presentó una latencia menor y un throughput superior a la inferencia en CPU para la resolución de 128 × 128 píxeles.
Por otra parte, el procesamiento paralelo permitió reducir el tiempo de extracción de características. La configuración de cuatro procesos obtuvo el menor tiempo, aunque la eficiencia fue inferior a la de dos procesos.
En conjunto, los resultados muestran que la calidad de clasificación y el rendimiento computacional dependen tanto del método utilizado como de la configuración de ejecución.
6.8. Evaluación del modelo exportado a ONNX
Se evaluó el rendimiento de MobileNetV2 después de su exportación al formato ONNX, utilizando ONNX Runtime como entorno de ejecución. En primer lugar, se verificó la equivalencia de las predicciones entre PyTorch y ONNX Runtime mediante un conjunto de imágenes de prueba. La validación permitió comprobar que ambas implementaciones generaron resultados coincidentes en las 50 imágenes utilizadas para esta comprobación. Posteriormente, se comparó el tiempo de ejecución de ONNX Runtime frente a PyTorch en CPU. Los resultados mostraron que ONNX Runtime alcanzó una aceleración aproximada de 6,13 veces respecto a la ejecución de PyTorch en CPU.
Estos resultados evidencian que la exportación a ONNX puede constituir una alternativa para optimizar la ejecución del modelo, manteniendo la equivalencia de las predicciones en las pruebas realizadas.
7. Discusión:
7.1. Comparación entre visión clásica y aprendizaje profundo
Los resultados del escenario B muestran una diferencia en la capacidad de clasificación de los dos enfoques evaluados. Random Forest obtuvo una exactitud del 93,89 %, mientras que MobileNetV2 alcanzó el 100 % con resolución de 128 × 128 píxeles. Una posible explicación de esta diferencia está relacionada con la forma en que cada método representa las imágenes. Random Forest utiliza características previamente definidas mediante HOG, LBP y HSV. Por su parte, MobileNetV2 aprende representaciones visuales a partir de los datos durante el entrenamiento. En este experimento, la combinación de características manuales permitió clasificar correctamente la mayoría de las imágenes, pero presentó dificultades en las categorías pimiento rojo y fresa. MobileNetV2 obtuvo resultados superiores en esas condiciones de evaluación. Sin embargo, esta conclusión debe limitarse a las cinco categorías y a las condiciones de orientación utilizadas en el proyecto.
7.2. Comportamiento de los escenarios experimentales
El escenario A obtuvo una exactitud del 100 % con Random Forest. No obstante, las imágenes de entrenamiento y prueba contenían orientaciones intercaladas. En el escenario B, las imágenes de prueba corresponden a orientaciones no incluidas en el entrenamiento. Bajo esta condición, la exactitud de Random Forest disminuyó al 93,89 %. Esta diferencia evidencia que el escenario B representa una evaluación más exigente para analizar el comportamiento del clasificador ante orientaciones no vistas. Por tanto, el resultado del escenario A no debe utilizarse como equivalente al resultado del escenario B ni como evidencia de que el modelo mantiene el mismo rendimiento ante orientaciones desconocidas.
7.3. Comportamiento del procesamiento paralelo
La extracción paralela de características redujo el tiempo de ejecución respecto a la modalidad secuencial. La configuración de dos procesos alcanzó una eficiencia del 91,23 %, mientras que cuatro procesos alcanzaron una eficiencia del 49,79 %. La disminución de eficiencia al utilizar cuatro procesos puede relacionarse con los costos adicionales de creación y administración de procesos, distribución de tareas y coordinación de resultados. Además, el problema experimental tiene un tamaño limitado: se procesaron 360 imágenes. En este contexto, los costos adicionales del paralelismo pueden representar una proporción relevante del tiempo total. Por ello, aunque cuatro procesos obtuvieron el menor tiempo, dos procesos presentaron una utilización más eficiente de los recursos disponibles.
7.4. Comportamiento entre CPU y GPU
MobileNetV2 obtuvo una latencia media de 19,294 ms en CPU y 7,828 ms en GPU, utilizando resolución de 128 × 128 píxeles. La ejecución en GPU presentó una latencia menor y un throughput superior, manteniendo la misma exactitud de clasificación. Estos resultados muestran que la GPU fue una alternativa más rápida para la inferencia en las condiciones evaluadas. Sin embargo, el notebook no separa el tiempo de transferencia de datos del tiempo de cómputo de la GPU. Por tanto, no es posible determinar cuánto tiempo correspondió exclusivamente al procesamiento del modelo y cuánto a otras operaciones de la evaluación. Esta limitación debe considerarse al interpretar la comparación entre CPU y GPU.
7.5. Efecto de la resolución de entrada
La reducción de la resolución de 128 × 128 a 64 × 64 píxeles permitió disminuir la latencia media de inferencia en CPU. No obstante, la exactitud también presentó una pequeña disminución, pasando del 100 % al 99,44 %. Esto evidencia que la resolución de entrada influye tanto en el tiempo de procesamiento como en la calidad de clasificación. La configuración de 64 × 64 puede resultar conveniente cuando se prioriza reducir el tiempo de procesamiento y se acepta una pequeña disminución en la exactitud. En cambio, la resolución de 128 × 128 obtuvo los mejores resultados de clasificación en el experimento.
7.6. Limitaciones del experimento
El proyecto presenta algunas limitaciones que deben considerarse al interpretar los resultados:
Se utilizaron únicamente cinco categorías de las cien disponibles en COIL-100.
La evaluación se concentró principalmente en variaciones de orientación, sin analizar sistemáticamente cambios de iluminación, oclusiones, fondos complejos o ruido.
El escenario A y el escenario B utilizan diferentes distribuciones de entrenamiento y prueba, por lo que sus resultados no son directamente equivalentes.
MobileNetV2 se evaluó en el escenario B, por lo que no se dispone de una comparación entre ambos modelos en el escenario A.
El conjunto de entrenamiento de MobileNetV2 no incluyó un conjunto de validación independiente registrado en el notebook.
No se registró por separado el tiempo de transferencia de datos hacia la GPU.
Las mediciones de latencia de Random Forest y MobileNetV2 incluyen operaciones diferentes, por lo que la comparación directa de sus tiempos debe realizarse con precaución.
El benchmark paralelo se realizó sobre 360 imágenes y en un entorno con cuatro CPU disponibles. Los resultados podrían variar con conjuntos de mayor tamaño o diferente hardware.
No se registró el modelo exacto de CPU y GPU ni las versiones completas de las bibliotecas utilizadas.
Estas limitaciones impiden generalizar los resultados a cualquier conjunto de imágenes o entorno computacional.
7.7. Evaluación de ONNX y presentación mediante dashboard web
La exportación de MobileNetV2 al formato ONNX permitió evaluar una alternativa de ejecución frente a PyTorch en CPU. La aceleración aproximada de 6,13 veces obtenida mediante ONNX Runtime muestra una mejora en el tiempo de ejecución bajo las condiciones experimentales evaluadas. Asimismo, la comprobación de las predicciones permitió verificar la equivalencia entre ambas implementaciones en las imágenes utilizadas para la validación.
Por otro lado, el dashboard web complementa la implementación al proporcionar una interfaz para interactuar con el sistema y presentar sus resultados. De esta manera, el proyecto no se limitó al desarrollo y evaluación de modelos, sino que también incorporó una herramienta para demostrar su funcionamiento.
8. Conclusiones
Se implementaron dos enfoques de clasificación de imágenes: uno basado en visión clásica mediante Random Forest y otro basado en aprendizaje profundo mediante MobileNetV2. Ambos fueron evaluados utilizando cinco categorías del dataset COIL-100.
El enfoque clásico obtuvo 100 % de accuracy en el escenario intercalado y 93.89 % en el escenario de ángulos no vistos. Esta diferencia evidencia que su rendimiento depende de las características visuales disponibles durante el entrenamiento.
MobileNetV2 alcanzó 100 % de accuracy en el escenario B con resolución de 128 × 128 píxeles, tanto en CPU como en GPU. Este resultado fue superior al obtenido por Random Forest en el mismo escenario.
La inferencia de MobileNetV2 en GPU registró una latencia media aproximada de 7.83 ms y un throughput de 127.75 imágenes por segundo, superando el rendimiento medido en CPU.
La extracción paralela de características permitió reducir el tiempo de ejecución. Con cuatro procesos se obtuvo un speedup aproximado de 1.99×, aunque la eficiencia disminuyó a 49.79 %.
El enfoque clásico presentó un menor consumo de memoria RAM que MobileNetV2 en las implementaciones evaluadas. Esto representa una ventaja cuando los recursos de memoria son limitados.
En términos generales, MobileNetV2 mostró una mayor capacidad de clasificación en el escenario experimental más exigente, mientras que Random Forest constituyó una alternativa de menor consumo de memoria y con resultados competitivos en la partición intercalada.
La exportación de MobileNetV2 al formato ONNX permitió evaluar una alternativa de inferencia mediante ONNX Runtime, obteniéndose una aceleración aproximada de 6,13 veces respecto a PyTorch en CPU, manteniendo la equivalencia de las predicciones en las pruebas de validación realizadas.
La evaluación de diferentes tamaños de lote permitió incorporar el análisis de la relación entre la cantidad de imágenes procesadas conjuntamente y el rendimiento computacional del modelo.
La implementación del dashboard web permitió complementar el sistema de clasificación mediante una interfaz de interacción y presentación de resultados.
En respuesta a la pregunta de investigación, los resultados muestran que MobileNetV2 presentó una mayor calidad de clasificación en el escenario de ángulos no vistos, mientras que Random Forest constituyó una alternativa con menor consumo de memoria RAM y resultados competitivos cuando las orientaciones estuvieron intercaladas. Por tanto, la elección del método depende del equilibrio requerido entre exactitud, consumo de recursos y tiempo de ejecución.



9. Referencias bibliográficas

Sucar, L. E., & Gómez, G. (2004). Visión computacional. Instituto Nacional de Astrofísica, Óptica y Electrónica. https://ccc.inaoep.mx/~esucar/Libros/vision-sucar-gomez.pdf
S. A. Nene, S. K. Nayar y H. Murase, “Columbia Object Image Library (COIL-100)”, Technical Report CUCS-006-96, 1996. https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR), 886–893. https://doi.org/10.1109/CVPR.2005.177
Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971–987. https://doi.org/10.1109/TPAMI.2002.1017623
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4510–4520. https://doi.org/10.1109/CVPR.2018.00474
Sural, S., Qian, G., & Pramanik, S. (2002). Segmentation and histogram generation using the HSV color space for image retrieval. Proceedings. International Conference on Image Processing, 2, 589–592. https://doi.org/10.1109/ICIP.2002.1040019
Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.


