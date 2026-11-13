# Machine Learning — Práctica 1: Enfermedad cardiovascular

Máster Universitario en Inteligencia Artificial, Universidad de Murcia. Curso 2026/2027.

## Autores

- Daniel Alexander Yearwood Agames — dalexander.yearwood@um.es
- Mario Castillo Gutiérrez — mario.c@um.es

## Descripción del problema

A partir de una cohorte clínica de 2.254 pacientes adultos con 27 variables sociodemográficas, clínicas, analíticas y de estilo de vida, se busca clasificar a cada paciente en uno de tres niveles de riesgo cardiovascular:

| Clase | Significado |
|-------|-------------|
| 0 | Riesgo bajo |
| 1 | Riesgo moderado |
| 2 | Riesgo alto |

## Objetivos

1. Análisis exploratorio del dataset e identificación de sus problemáticas.
2. Aplicación de al menos tres técnicas de preprocesamiento para resolverlas.
3. Entrenamiento y ajuste de hiperparámetros de al menos cinco clasificadores:
   - Dos modelos base (lineales o sencillos).
   - Tres modelos de ensemble.
4. Evaluación con un esquema de validación robusto y al menos tres métricas, justificando la más relevante desde el punto de vista clínico.
5. Conclusiones sobre el mejor modelo (rendimiento, complejidad, interpretabilidad y robustez).
6. Opcional: comparación de modelos mediante tests estadísticos.

## Estructura del repositorio

```
ml-practica1-cardiovascular/
├── data/                 # Dataset (enfermedad_cardiovascular.csv)
├── docs/
│   └── bitacora.md       # Evolución del proyecto y decisiones de diseño
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb
├── results/              # Figuras y tablas generadas
├── src/
│   └── config.py         # Rutas, semilla y parámetros comunes
├── requirements.txt
└── README.md
```

## Reproducción

Requisito: Python 3.12.

```bash
# Crear y activar el entorno virtual
python3.12 -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Abrir los notebooks
jupyter lab
```

Copiar `enfermedad_cardiovascular.csv` en la carpeta `data/` antes de ejecutar los notebooks.

Todos los procesos aleatorios usan la semilla definida en `src/config.py` (`SEED = 42`).

## Entregables

- Repositorio GitHub (este repositorio).
- Memoria técnica (plantilla IEEE, máximo 15 páginas).
- Vídeo explicativo (máximo 7 minutos).
- Informe de uso ético y responsable.
