# ¿Reconoce el modelo un objeto desde un ángulo que nunca vio?

Cinco objetos de COIL-100 giran sobre una plataforma y se fotografían cada 5°. Se compara un Random Forest con características clásicas frente a MobileNetV2, y se mide cuánto aceleran la GPU, el procesamiento paralelo y ONNX.

EntrenamientoPrueba

## Calidad de clasificación

En el escenario A, las vistas de prueba quedan entre vistas de entrenamiento muy parecidas, así que acertar es fácil. En el B, los ángulos de 180° a 355° no aparecen en el entrenamiento: es la prueba real de generalización.

### Random Forest, escenario B: recall por objeto

El objeto 63 absorbe falsos positivos (precisión 0,766) y el 75 pierde un 30,6 % de sus vistas.

### Efecto de la resolución en MobileNetV2

A 64×64 el entrenamiento es 3,3× más rápido, pero se pierden 4 imágenes de 180.

## Velocidad de inferencia

Latencia media por imagen. Las tres primeras filas incluyen leer la imagen y prepararla; las dos últimas miden solo el modelo sobre tensores ya listos, por eso no son comparables con las anteriores.

**6,13×**ONNX frente a PyTorch en CPU

**−83,7 %**latencia con ONNX

**2,3×**GPU frente a CPU, con carga de imagen

## Tamaño de lote

Procesar varias imágenes juntas aprovecha mejor el hardware, sobre todo en GPU. La precisión se mantiene en 100 % en todos los casos.

## Paralelización de características

Extraer color, HOG y LBP de las 360 imágenes del escenario B con varios procesos. Con 4 hilos lógicos (2 núcleos físicos) el tiempo casi no mejora al pasar de 2 a 4 procesos.

**1,80×**con 2 procesos

**1,95×**con 4 procesos

La eficiencia cae de 90 % a 49 %. Los resultados son idénticos a la versión secuencial.

## Prueba con cámara

Clasifica una foto con tu propio modelo exportado. Carga el archivo **mobilenet_128_scenario_B.onnx** de la carpeta results del notebook; todo se procesa en tu navegador y ninguna imagen sale de tu dispositivo. COIL-100 usa objetos sobre fondo negro, así que obtendrás mejores resultados con un fondo oscuro liso.

Subir foto

Cargar modelo .onnx

Sin modelo cargado.

## Exportación a ONNX

El modelo MobileNetV2 de 128×128 se exportó con tamaño de lote dinámico y se comparó con PyTorch sobre 50 imágenes de prueba.

Tamaño del archivo

8,48 MB

Opset

18

Entrada

\[batch, 3, 128, 128\]

Salida

\[batch, 5\]

Parámetros

2.230.277

Coincidencia de predicciones

100 %

Diferencia máxima de logits

1,76 × 10⁻⁵

Precisión en la muestra

100 %

Latencia p95

3,40 ms (PyTorch: 17,70 ms)

## Entorno experimental

Plataforma

Kaggle, Linux x86_64

CPU

2 núcleos físicos, 4 hilos lógicos

RAM

31,35 GB

GPU

2 × Tesla T4 (CUDA 13.0)

Software

PyTorch 2.11, scikit-learn 1.6.1, OpenCV 4.14, NumPy 2.1.3

Configuración

8 épocas, lote de 32, tasa de aprendizaje 1e-4, 100 árboles, semilla 42

Objetos

43, 63, 68, 74 y 75 (72 vistas cada uno)

Datos tomados de las salidas del notebook. Dataset: COIL-100 (7.200 imágenes, 100 objetos).