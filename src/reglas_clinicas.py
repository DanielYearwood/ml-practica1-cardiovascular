"""Reglas para detectar valores imposibles en las variables clínicas.

Marcamos como imposibles los valores que no pueden darse en un paciente adulto
o que son errores claros (negativos, más de 7 días a la semana, etc.). Los
valores extremos que sí son posibles no se marcan aquí, se tratan como outliers.

Los límites son fijos y no se calculan a partir de los datos, así que se pueden
usar dentro del pipeline sin que haya fuga de información.
"""

import numpy as np
import pandas as pd

# (mínimo, máximo) de cada variable. None si no hay límite por ese lado.
RANGOS_CLINICOS = {
    "edad": (18, 110),                        # la cohorte es de adultos
    "presion_sistolica": (60, 260),           # mmHg
    "presion_diastolica": (30, 150),          # mmHg
    "colesterol_total": (70, 600),            # mg/dL
    "colesterol_hdl": (10, 150),              # mg/dL
    "colesterol_ldl": (10, 400),              # mg/dL
    "trigliceridos": (20, 2000),              # mg/dL
    "glucosa_ayunas": (40, 600),              # mg/dL
    "indice_masa_corporal": (12, 70),         # kg/m2
    "frecuencia_cardiaca_reposo": (30, 220),  # latidos por minuto
    "horas_sueno_diarias": (0, 24),
    "consumo_alcohol_semanal": (0, None),     # no puede ser negativo
    "anos_fumador": (0, None),                # además tiene que ser <= edad - 10
    "proteina_c_reactiva": (0, None),         # mg/L
    "acido_urico": (1, 15),                   # mg/dL
    "clearance_creatinina": (5, 200),         # mL/min
    "frecuencia_ejercicio_semanal": (0, 7),   # días a la semana
    "perimetro_cintura": (50, 200),           # cm
    "hemoglobina_glicosilada": (3.5, 20),     # %
}

# edad mínima a la que suponemos que se puede empezar a fumar
EDAD_MIN_INICIO_TABACO = 10


def mascara_valores_imposibles(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame de True/False con True en los valores imposibles."""
    mascara = pd.DataFrame(False, index=df.index, columns=list(RANGOS_CLINICOS))
    for col, (minimo, maximo) in RANGOS_CLINICOS.items():
        if minimo is not None:
            mascara[col] |= df[col] < minimo
        if maximo is not None:
            mascara[col] |= df[col] > maximo

    # si la diastólica es >= que la sistólica no sabemos cuál está mal,
    # así que marcamos las dos
    presion_invertida = df["presion_diastolica"] >= df["presion_sistolica"]
    mascara["presion_sistolica"] |= presion_invertida
    mascara["presion_diastolica"] |= presion_invertida

    # no se pueden llevar más años fumando que los vividos desde los 10 años
    mascara["anos_fumador"] |= df["anos_fumador"] > df["edad"] - EDAD_MIN_INICIO_TABACO
    return mascara


def anular_valores_imposibles(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de df con los valores imposibles cambiados a NaN."""
    df = df.copy()
    mascara = mascara_valores_imposibles(df)
    for col in mascara.columns:
        df.loc[mascara[col], col] = np.nan
    return df
