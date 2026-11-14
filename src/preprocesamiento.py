"""Preprocesamiento de los datos para los modelos.

Todo va dentro de un Pipeline de imbalanced-learn para que la imputación, el
escalado y SMOTE se ajusten solo con los datos de entrenamiento de cada fold
y no haya fuga de información hacia validación o test.

Uso básico:

    from src.preprocesamiento import construir_pipeline
    pipe = construir_pipeline(LogisticRegression(), imputacion="knn",
                              escalado="yeojohnson", balanceo="smotenc")
    pipe.fit(X_train, y_train)

Pasos del pipeline:
    1. anular       -> valores imposibles a NaN (reglas de src/reglas_clinicas.py)
    2. imputar      -> escalado + imputación de las numéricas
    3. smote        -> SMOTE-NC (solo si balanceo="smotenc")
    4. codificar    -> ordinal para nivel_actividad y one-hot para las nominales
    5. modelo
"""

from itertools import product

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   PowerTransformer, StandardScaler)
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (BINARY_FEATURES, CATEGORICAL_FEATURES,
                        NUMERIC_FEATURES, SEED)
from src.reglas_clinicas import anular_valores_imposibles

# orden natural de la variable ordinal (de menos a más actividad)
ORDEN_NIVEL_ACTIVIDAD = ["Sedentario", "Ligeramente activo",
                         "Moderadamente activo", "Muy activo"]
VARIABLE_ORDINAL = ["nivel_actividad"]
VARIABLES_NOMINALES = [c for c in CATEGORICAL_FEATURES if c not in VARIABLE_ORDINAL]

OPCIONES_IMPUTACION = ["mediana", "knn"]
OPCIONES_ESCALADO = ["estandar", "yeojohnson"]
OPCIONES_BALANCEO = ["ninguno", "class_weight", "smotenc"]


class AnuladorValoresImposibles(BaseEstimator, TransformerMixin):
    """Pasa a NaN los valores imposibles. No aprende nada de los datos."""

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        return anular_valores_imposibles(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def _escalador(escalado):
    if escalado == "estandar":
        return StandardScaler()
    if escalado == "yeojohnson":
        # Yeo-Johnson también estandariza (standardize=True por defecto)
        return PowerTransformer(method="yeo-johnson")
    raise ValueError(f"escalado no válido: {escalado}. Opciones: {OPCIONES_ESCALADO}")


def _imputador(imputacion):
    if imputacion == "mediana":
        return SimpleImputer(strategy="median")
    if imputacion == "knn":
        return KNNImputer(n_neighbors=5)
    raise ValueError(f"imputacion no válida: {imputacion}. Opciones: {OPCIONES_IMPUTACION}")


def construir_preprocesador(imputacion="mediana", escalado="estandar", balanceo="ninguno",
                            onehot_drop=None):
    """Devuelve la lista de pasos de preprocesamiento (sin el modelo).

    imputacion: "mediana" o "knn"
    escalado: "estandar" (z-score) o "yeojohnson"
    balanceo: "ninguno", "class_weight" o "smotenc". Solo "smotenc" añade un
              paso aquí; "class_weight" se aplica en el modelo.
    onehot_drop: None o "first" (quitar una columna dummy, útil en modelos lineales)
    """
    # primero escalamos y luego imputamos: así KNN calcula las distancias con
    # todas las variables en la misma escala. Los dos escaladores ignoran los
    # NaN al ajustarse y los dejan pasar.
    numericas = Pipeline([
        ("escalar", _escalador(escalado)),
        ("imputar", _imputador(imputacion)),
    ])
    imputar = ColumnTransformer(
        [("num", numericas, NUMERIC_FEATURES),
         ("resto", "passthrough", BINARY_FEATURES + CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    codificar = ColumnTransformer(
        [("ordinal", OrdinalEncoder(categories=[ORDEN_NIVEL_ACTIVIDAD]), VARIABLE_ORDINAL),
         ("onehot", OneHotEncoder(drop=onehot_drop, sparse_output=False), VARIABLES_NOMINALES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    pasos = [("anular", AnuladorValoresImposibles()), ("imputar", imputar)]
    if balanceo == "smotenc":
        # SMOTE-NC va antes de codificar para que trate las binarias y las
        # categóricas como categorías y no invente valores como 0.4
        pasos.append(("smote", SMOTENC(categorical_features=BINARY_FEATURES + CATEGORICAL_FEATURES,
                                       random_state=SEED)))
    elif balanceo not in OPCIONES_BALANCEO:
        raise ValueError(f"balanceo no válido: {balanceo}. Opciones: {OPCIONES_BALANCEO}")
    pasos.append(("codificar", codificar))
    return pasos


def construir_pipeline(modelo, imputacion="mediana", escalado="estandar", balanceo="ninguno",
                       onehot_drop=None):
    """Pipeline completo: preprocesamiento + modelo.

    Con balanceo="class_weight" se pone class_weight="balanced" en el modelo.
    Si el modelo no tiene ese parámetro (GradientBoosting, XGBoost...) da error:
    en ese caso hay que usar balanceo="ninguno" y pasar los pesos al entrenar:
        pipe.fit(X, y, modelo__sample_weight=pesos_balanceados(y))
    """
    if balanceo == "class_weight":
        if "class_weight" not in modelo.get_params():
            raise ValueError(f"{type(modelo).__name__} no tiene class_weight; "
                             "usa modelo__sample_weight=pesos_balanceados(y) al hacer fit")
        modelo.set_params(class_weight="balanced")
    pasos = construir_preprocesador(imputacion, escalado, balanceo, onehot_drop)
    pasos.append(("modelo", modelo))
    return Pipeline(pasos)


def pesos_balanceados(y):
    """Peso de cada muestra inversamente proporcional a la frecuencia de su clase."""
    return compute_sample_weight(class_weight="balanced", y=y)


def todas_las_configuraciones():
    """Lista con todas las combinaciones imputación x escalado x balanceo (12)."""
    return [{"imputacion": i, "escalado": e, "balanceo": b}
            for i, e, b in product(OPCIONES_IMPUTACION, OPCIONES_ESCALADO, OPCIONES_BALANCEO)]
