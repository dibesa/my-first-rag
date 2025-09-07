# my-first-rag
Evaluacion Técnica AI Engineer MasOrange

## Ejecución de los ejercicios
Es necesario contar con una instalación de python y de jupyter, yo recomiendo usar [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

### Preparación del entorno
Lo primero que hay que hacer es descargar el proyecto ejecutando en un terminal:
```
git clone https://github.com/dibesa/my-first-rag.git
```

Puedes crear antes un virtualenv con conda si no quieres hacer la instalación en el entorno base (opcional):
```
conda create -n rag_env
conda activate rag_env
```

Instalar las librerías necesarias ejecutando:
```
cd my-first-rag
pip install -r requirements.txt
```

### Ejercicio 1, 2, 3
Los primeros tres ejercicios se pueden ejecutar en Jupyter Notebook. Simplemente ejecutar:
```
jupyter notebook
```
Se abrirá una ventana del navegador que tengas configurado por defecto mostrando el sistema de archivos. Los archivos .ipynb tienen el nombre de cada ejercicio.

### Ejercicio 4
Ejecutar el comando:
```
python chatbot.py
```
Esperar a que cargue la BBDD y aparecerá un prompt para que puedas indicar tu pregunta. Si desea salir escriba 'exit' o cierre el terminal.
