# aapi-cbir
AAPI 2024 CBIR  
Grupo B: Alejandro Pastor Membrado, Ángel Romero Huici

## Información
Las imágenes utilizadas para este proyecto proceden del conjunto de datos https://www.kaggle.com/datasets/elkamel/corel-images  
y ya han sido tratadas para el funcionamiento del sistema.  

Se incluye en el notebook la transformación del conjunto original de datos a los utilizados para el proyecto aunque no esté incluido el conjunto de datos original.

## Contenidos
- app.py  
  - Código de ejecución de la interfaz

- /project 
  - Índices FAISS para cada uno de los 5 extractores
  - /images con las 900 imagenes de base de datos
  - /queries con 100 imagenes propuestas para realizar las búsquedas
  - projectfaiss.ipynb con la implementación del código de los extractores, indexación y evaluación
 
## Ejecución
- app.py, se ejecuta en http://localhost:8501  
    *streamlit run app.py*
- projectfaiss.ipynb, Jupyter Notebook
