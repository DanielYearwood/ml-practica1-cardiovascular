# Bitácora del proyecto

Registro de la evolución del planteamiento, las decisiones de diseño y los avances de la práctica.

## Organización del trabajo

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Análisis exploratorio e identificación de problemáticas | Completada |
| 2 | Preprocesamiento (mínimo 3 técnicas) | Completada |
| 3 | Modelos base (mínimo 2) | Pendiente |
| 4 | Modelos de ensemble (mínimo 3) | Pendiente |
| 5 | Validación, métricas y ajuste de hiperparámetros | En curso |
| 6 | Conclusiones y tests estadísticos (opcional) | Pendiente |
| 7 | Memoria, vídeo e informe de uso ético | Pendiente |

## Registro

### Inicio del proyecto

- Creación del repositorio y de la estructura de carpetas.
- Definición del entorno (`requirements.txt`, Python 3.12) y de la semilla común (`src/config.py`).
- Notebook inicial de carga y primera inspección del dataset.

### Fase 1 · Análisis exploratorio (rama `feature/3-analisis-exploratorio`)

Hemos ampliado el notebook `01_analisis_exploratorio.ipynb` para revisar la calidad de los datos. Además de lo básico (nulos, duplicados, distribución de la clase, outliers y correlaciones) hemos añadido:

- Comprobación de que las columnas coinciden con el enunciado.
- Test chi-cuadrado para ver si los nulos dependen de la clase.
- Detección de valores imposibles con rangos clínicos. Las reglas están en `src/reglas_clinicas.py` para poder reutilizarlas en el preprocesamiento.
- Valores recortados en el percentil 99.
- Varianza casi cero (criterio del tema 1).
- Revisión de las categóricas por si había errores de escritura.
- Incoherencias lógicas entre variables.
- Relación de cada variable con la clase (Kruskal-Wallis y chi-cuadrado con tamaño del efecto).

Problemas encontrados (están explicados con cifras en la sección 12 del notebook):

1. Nulos en 4 variables (23,6 % de las filas), que no dependen de la clase.
2. 392 valores imposibles en 325 pacientes (alcohol negativo, diastólica >= sistólica, más de 7 días de ejercicio...).
3. Variables recortadas en el percentil 99.
4. Outliers y asimetría.
5. Escalas muy distintas.
6. Clases desbalanceadas (50/30/20) con la de riesgo alto como minoritaria.
7. Muchas incoherencias lógicas (HDL + LDL > total, no fumadores con años de fumador, etc.).
8. Categóricas que hay que codificar y mezcla de tipos de variables.
9. Casi ninguna correlación entre variables y efectos pequeños con la clase.

No hay duplicados, ni variables con varianza casi cero, ni errores en las categóricas.

Decisiones que hemos tomado:

- Solo marcamos como imposibles los valores que no pueden darse o que son errores claros. Los valores extremos pero posibles los dejamos.
- El alcohol negativo lo pasamos a nulo en vez de a 0, porque poner 0 sería inventarnos que el paciente no bebe.
- Las incoherencias lógicas no las corregimos porque no sabemos qué variable está mal. Solo anulamos la diastólica >= sistólica y los años fumando > edad - 10. Pensamos que el dataset es sintético y lo comentaremos en la memoria.
- No eliminamos filas, porque con los valores imposibles como nulos se perdería el 34,3 % de los pacientes. Habrá que imputar.

Las figuras y tablas están en `results/`.

### Fase 2 · Preprocesamiento (rama `feature/4-preprocesamiento`)

Hemos creado `src/preprocesamiento.py` con todo el preprocesamiento dentro de un `Pipeline` de imbalanced-learn, y el notebook `02_preprocesamiento.ipynb` donde explicamos y comprobamos cada técnica.

Técnicas aplicadas:

- T1. Valores imposibles a NaN (con las reglas clínicas) e imputación. Comparamos mediana y KNN (5 vecinos). No añadimos indicador de nulos porque en el EDA vimos que no dependen de la clase.
- T2. Escalado. Comparamos z-score y Yeo-Johnson. Con Yeo-Johnson la asimetría de PCR, triglicéridos, colesterol total, sistólica y ácido úrico baja de 1,3-1,6 a casi 0.
- T3. Balanceo. Comparamos no hacer nada, `class_weight="balanced"` y SMOTE-NC. Usamos SMOTE-NC y no SMOTE porque hay binarias y categóricas. Para los modelos sin `class_weight` (GradientBoosting, XGBoost) hay que pasar `sample_weight` con la función `pesos_balanceados`.
- T4. Codificación. Ordinal para `nivel_actividad` y one-hot para `tipo_alimentacion` y `antecedentes_familiares_tipo`.

Decisiones:

- En el pipeline primero se escala y luego se imputa, para que KNN calcule las distancias con todas las variables en la misma escala.
- SMOTE-NC va antes de codificar las categóricas, para que las trate como categorías.
- Al estar todo dentro del pipeline, SMOTE solo se aplica al train de cada fold en la validación cruzada y la validación se hace con pacientes reales.
- Salen 12 combinaciones de preprocesamiento (2 imputaciones x 2 escalados x 3 balanceos). Hemos comprobado que todas funcionan con una regresión logística; la comparación con todos los modelos va en el notebook 03.

Cómo usarlo para entrenar un modelo:

```python
from src.preprocesamiento import construir_pipeline, todas_las_configuraciones
pipe = construir_pipeline(modelo, imputacion="knn", escalado="yeojohnson", balanceo="smotenc")
```

### Fase 3 · Validación y ajuste de hiperparámetros (rama `feature/7-validacion`)

Hemos creado `src/validacion.py` con el esquema de validación que van a usar todos los modelos, y el notebook `03_validacion.ipynb` donde lo explicamos y lo probamos con una regresión logística.

Esquema:

- Partición train/test 80/20 estratificada con la semilla de `config.py`. El test no se usa hasta el final.
- Con el train, `GridSearchCV` (o `RandomizedSearchCV` en los modelos lentos) con `StratifiedKFold` de 5 particiones, para cada una de las 12 configuraciones de preprocesamiento.
- Para comparar los mejores modelos, validación cruzada 5x3 con las mismas particiones para todos. Esto nos sirve también para los tests estadísticos.
- Al final, cada modelo elegido se entrena con todo el train y se evalúa en test.

Métricas: F1 macro, balanced accuracy, recall de riesgo alto y accuracy. Los hiperparámetros se eligen por F1 macro.

Decisiones:

- La métrica más importante desde el punto de vista clínico es el recall de riesgo alto, porque no detectar a un paciente de riesgo alto es el error más grave. No la usamos para elegir hiperparámetros porque un modelo que diga "alto" a todo el mundo tendría recall 1. Elegimos con F1 macro y miramos el recall de riesgo alto al comparar.
- Como todo el preprocesamiento está en el pipeline, en cada fold SMOTE-NC solo se aplica al train y la validación se hace con pacientes reales. Lo hemos comprobado en el notebook. Es lo mismo que aclaró la profesora en el anuncio del 13/11.
- Para los modelos sin `class_weight` (GradientBoosting, XGBoost) la configuración `class_weight` se hace con pesos por muestra.
- No usamos validación cruzada anidada ni leave-one-out porque con 12 configuraciones y varios modelos tardaría demasiado.

Primeros resultados con la logística: el F1 macro está entre 0,62 y 0,64 en todas las configuraciones, pero el recall de riesgo alto pasa de 0,66 sin balancear a 0,71-0,76 con `class_weight` o SMOTE-NC. La clase moderado es la que peor se predice.
