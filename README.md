Proyecto que contiene la aplicacion desarrollada en FastAPI la cual se encargara de la aplicación del modelo entrenado de Deep Learning para detección de lengua de señas

### Requisitos previos
Tener un entorno virtual de Python 3.9

# Instalacion y ejecucion del proyecto

### Instalacion de paquetes

```
$ pip install -r requirements.txt
```

### Ejecucion de la aplicacion FastAPI con uvicorn para entorno de desarrollo

```
$ uvicorn main:app --reload
```
### Ejecucion de la aplicacion FastAPI con uvicorn para entorno de producción

```
$ uvicorn main:app --host 0.0.0.0 --port 8000
```