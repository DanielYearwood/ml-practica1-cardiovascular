"""Esquema de validación común para todos los modelos.

1. Separamos un 20 % de test estratificado que no se toca hasta el final.
2. Con el 80 % restante hacemos GridSearchCV con StratifiedKFold de 5 particiones,
   para cada una de las 12 configuraciones de preprocesamiento.
3. Elegimos los hiperparámetros por F1 macro y guardamos también balanced
   accuracy, recall de la clase de riesgo alto y accuracy.
4. Para comparar los mejores modelos entre sí (y para los tests estadísticos)
   repetimos la validación cruzada 3 veces con las mismas particiones para todos.
5. El test solo se usa al final con los modelos ya elegidos.

Uso básico:

    from src.validacion import cargar_datos, dividir_train_test, ajustar_modelo
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_train_test(X, y)
    tabla, busquedas = ajustar_modelo("logistica", LogisticRegression(max_iter=2000),
                                      {"C": [0.1, 1, 10]}, X_train, y_train)
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             make_scorer, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, train_test_split)

from src.config import (CLASS_NAMES, CV_FOLDS, DATA_FILE, RESULTS_DIR, SEED,
                        TARGET, TEST_SIZE)
from src.preprocesamiento import (construir_pipeline, pesos_balanceados,
                                  todas_las_configuraciones)

CLASE_ALTO = 2
REPETICIONES = 3

# métricas que guardamos en todas las búsquedas
METRICAS = {
    "f1_macro": "f1_macro",
    "bal_acc": "balanced_accuracy",
    "recall_alto": make_scorer(recall_score, labels=[CLASE_ALTO], average="macro"),
    "accuracy": "accuracy",
}
# con esta se eligen los hiperparámetros: obliga a acertar las tres clases y
# no deja ganar a un modelo que diga "alto" a todo el mundo
METRICA_REFIT = "f1_macro"


def cargar_datos():
    df = pd.read_csv(DATA_FILE)
    return df.drop(columns=TARGET), df[TARGET]


def dividir_train_test(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def particiones_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def particiones_comparacion():
    # mismas particiones para todos los modelos -> resultados emparejados
    return RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=REPETICIONES,
                                   random_state=SEED)


def _tiene_class_weight(modelo):
    return "class_weight" in modelo.get_params()


def preparar_pipeline(modelo, imputacion, escalado, balanceo, onehot_drop=None):
    """Devuelve (pipeline, parámetros extra para fit).

    Si el modelo no tiene class_weight (GradientBoosting, XGBoost...), el
    balanceo "class_weight" se hace pasando pesos por muestra al entrenar.
    """
    modelo = clone(modelo)
    if balanceo == "class_weight" and not _tiene_class_weight(modelo):
        pipe = construir_pipeline(modelo, imputacion, escalado, "ninguno", onehot_drop)
        return pipe, {"usar_pesos": True}
    pipe = construir_pipeline(modelo, imputacion, escalado, balanceo, onehot_drop)
    return pipe, {"usar_pesos": False}


def _parametros_fit(extra, y):
    # los pesos se calculan con las etiquetas de entrenamiento; GridSearchCV y
    # cross_validate los recortan solos en cada fold
    if extra["usar_pesos"]:
        return {"modelo__sample_weight": pesos_balanceados(y)}
    return {}


def ajustar_modelo(nombre, modelo, rejilla, X_train, y_train, configuraciones=None,
                   n_iter=None, onehot_drop=None, n_jobs=-1, verbose=True):
    """Ajusta los hiperparámetros del modelo con cada configuración de preprocesamiento.

    rejilla: diccionario con los hiperparámetros del modelo sin prefijo,
             por ejemplo {"C": [0.1, 1, 10]}.
    n_iter: si se indica, se usa RandomizedSearchCV con ese número de pruebas
            (útil en los modelos lentos). Si no, GridSearchCV.

    Devuelve una tabla con una fila por configuración y un diccionario con
    las búsquedas ya entrenadas.
    """
    if configuraciones is None:
        configuraciones = todas_las_configuraciones()
    rejilla = {f"modelo__{k}": v for k, v in rejilla.items()}

    filas, busquedas = [], {}
    for conf in configuraciones:
        pipe, extra = preparar_pipeline(modelo, onehot_drop=onehot_drop, **conf)
        comun = dict(scoring=METRICAS, refit=METRICA_REFIT, cv=particiones_cv(),
                     n_jobs=n_jobs, return_train_score=True, error_score="raise")
        if n_iter is None:
            busqueda = GridSearchCV(pipe, rejilla, **comun)
        else:
            busqueda = RandomizedSearchCV(pipe, rejilla, n_iter=n_iter,
                                          random_state=SEED, **comun)

        inicio = time.perf_counter()
        busqueda.fit(X_train, y_train, **_parametros_fit(extra, y_train))
        segundos = time.perf_counter() - inicio

        i = busqueda.best_index_
        res = busqueda.cv_results_
        fila = {"modelo": nombre, **conf,
                "mejores_parametros": {k.replace("modelo__", ""): v
                                       for k, v in busqueda.best_params_.items()}}
        for m in METRICAS:
            fila[f"{m}_media"] = res[f"mean_test_{m}"][i]
            fila[f"{m}_std"] = res[f"std_test_{m}"][i]
        # diferencia train - validación para ver si hay sobreajuste
        fila["f1_macro_train"] = res["mean_train_f1_macro"][i]
        fila["sobreajuste"] = fila["f1_macro_train"] - fila["f1_macro_media"]
        fila["segundos"] = round(segundos, 1)
        filas.append(fila)
        busquedas[_clave(conf)] = busqueda

        if verbose:
            print(f"{nombre} | {_clave(conf):<32} F1 macro {fila['f1_macro_media']:.3f} "
                  f"(±{fila['f1_macro_std']:.3f})  recall alto {fila['recall_alto_media']:.3f}"
                  f"  [{segundos:.0f} s]")

    tabla = pd.DataFrame(filas).sort_values("f1_macro_media", ascending=False)
    return tabla.reset_index(drop=True), busquedas


def _clave(conf):
    return f"{conf['imputacion']}-{conf['escalado']}-{conf['balanceo']}"


def mejor_configuracion(tabla):
    """Fila con mejor F1 macro medio en validación cruzada."""
    return tabla.sort_values("f1_macro_media", ascending=False).iloc[0]


def pipeline_desde_fila(modelo, fila, onehot_drop=None):
    """Rehace el pipeline (sin entrenar) con la configuración y parámetros de una fila."""
    modelo = clone(modelo).set_params(**fila["mejores_parametros"])
    return preparar_pipeline(modelo, fila["imputacion"], fila["escalado"],
                             fila["balanceo"], onehot_drop)


def comparar_modelos(candidatos, X_train, y_train, n_jobs=-1):
    """Validación cruzada repetida (5x3) con las mismas particiones para todos.

    candidatos: diccionario {nombre: (modelo, fila de la tabla)}.
    Devuelve una tabla larga con una fila por modelo y partición, que sirve
    para los gráficos y para los tests estadísticos (Friedman, Wilcoxon).
    """
    cv = particiones_comparacion()
    filas = []
    for nombre, (modelo, fila) in candidatos.items():
        pipe, extra = pipeline_desde_fila(modelo, fila)
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=METRICAS,
                             n_jobs=n_jobs, params=_parametros_fit(extra, y_train),
                             error_score="raise")
        for k in range(len(res["fit_time"])):
            filas.append({"modelo": nombre, "particion": k,
                          **{m: res[f"test_{m}"][k] for m in METRICAS}})
    return pd.DataFrame(filas)


def entrenar_final(modelo, fila, X_train, y_train):
    """Entrena con todo el train el pipeline elegido."""
    pipe, extra = pipeline_desde_fila(modelo, fila)
    pipe.fit(X_train, y_train, **_parametros_fit(extra, y_train))
    return pipe


def evaluar_en_test(pipe, X_test, y_test):
    """Métricas en test y matriz de confusión. Solo para los modelos finales."""
    pred = pipe.predict(X_test)
    metricas = {
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "bal_acc": balanced_accuracy_score(y_test, pred),
        "recall_alto": recall_score(y_test, pred, labels=[CLASE_ALTO], average="macro"),
        "accuracy": accuracy_score(y_test, pred),
    }
    etiquetas = sorted(CLASS_NAMES)
    matriz = pd.DataFrame(confusion_matrix(y_test, pred, labels=etiquetas),
                          index=[f"real {CLASS_NAMES[c]}" for c in etiquetas],
                          columns=[f"pred {CLASS_NAMES[c]}" for c in etiquetas])
    informe = classification_report(y_test, pred, labels=etiquetas,
                                    target_names=[CLASS_NAMES[c] for c in etiquetas],
                                    digits=3)
    return metricas, matriz, informe


def guardar_tabla(tabla, nombre):
    ruta = RESULTS_DIR / f"{nombre}.csv"
    tabla.to_csv(ruta, index=False)
    return ruta
