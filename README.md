# Detección de Esquizofrenia mediante señales EEG - Proyecto Final

Este repositorio contiene la implementación del proyecto final para la materia **Introduction to Deep Learning**. El objetivo es demostrar el uso de arquitecturas de aprendizaje profundo para identificar biomarcadores de esquizofrenia en registros de EEG, comparando su rendimiento frente a modelos estadísticos tradicionales.

## Descripción del Proyecto

El proyecto aborda la clasificación binaria de pacientes con esquizofrenia versus controles sanos. Se utiliza un pipeline que va desde el preprocesamiento de señales crudas hasta la clasificación mediante una **Red Neuronal Densa (MLP)**. 

**Puntos clave:**
* **Preprocesamiento:** Limpieza de artefactos con ICA y ICLabel.
* **Extracción de Características:** Potencia Espectral Relativa (PSD) en 5 bandas (Delta, Theta, Alfa, Beta, Gamma).
* **Validación:** Esquema *Leave-One-Subject-Out* (LOSO) para garantizar que el modelo generalice a pacientes nuevos.
* **Diagnóstico:** Sistema de votación mayoritaria por sujeto.

---

## Guía de Ejecución

Para reproducir los resultados obtenidos, se deben ejecutar los archivos en el siguiente orden:

### 1. Limpieza y Preprocesamiento
Ejecutar el cuaderno:
* `Limpieza_automatica_datos.ipynb`
Este paso carga los archivos crudos (.edf), aplica filtros, realiza la limpieza mediante ICA (Análisis de Componentes Independientes) y segmenta la señal.

### 2. Generación del Dataset
Al finalizar la ejecución del notebook anterior, se generará en el directorio raíz el archivo:
* `datos_eeg_procesados.npz`
**Nota:** Este archivo es indispensable para correr los modelos de entrenamiento.

### 3. Entrenamiento de la Red Neuronal (Deep Learning)
Ejecutar el script:
* `model.py`
Este es el modelo principal. Realiza la extracción de características espectrales y entrena el Perceptrón Multicapa (MLP) bajo la validación LOSO. Genera las métricas de desempeño y gráficas de confianza. Para ejecutar el estudio de ablación, se debe ejecutar el script * `model_2.py`

### 4. Comparación con Modelo Base
Ejecutar el script:
* `comparacion_modelo_simple.py`
Este archivo entrena una **Regresión Logística** bajo las mismas condiciones. Se utiliza como baseline para demostrar que la complejidad no lineal de la red neuronal aporta una mejora sustancial en el diagnóstico.

---

## Requisitos
El entorno requiere las siguientes librerías instaladas:
* `numpy`
* `pandas`
* `tensorflow`
* `scikit-learn`
* `scipy`
* `mne`
* `matplotlib`
* `seaborn`