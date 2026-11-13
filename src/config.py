"""Configuración común del proyecto."""

from pathlib import Path

# Semilla para garantizar la reproducibilidad
SEED = 42

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DATA_FILE = DATA_DIR / "enfermedad_cardiovascular.csv"

# Variable objetivo y clases
TARGET = "riesgo_cardiovascular"
CLASS_NAMES = {0: "Bajo", 1: "Moderado", 2: "Alto"}

# Tipos de variables según el diccionario del enunciado
NUMERIC_FEATURES = [
    "edad",
    "presion_sistolica",
    "presion_diastolica",
    "colesterol_total",
    "colesterol_hdl",
    "colesterol_ldl",
    "trigliceridos",
    "glucosa_ayunas",
    "indice_masa_corporal",
    "frecuencia_cardiaca_reposo",
    "horas_sueno_diarias",
    "consumo_alcohol_semanal",
    "anos_fumador",
    "proteina_c_reactiva",
    "acido_urico",
    "clearance_creatinina",
    "frecuencia_ejercicio_semanal",
    "perimetro_cintura",
    "hemoglobina_glicosilada",
]

BINARY_FEATURES = [
    "fumador_actual",
    "hipertension_previa",
    "diabetes_mellitus",
    "tratamiento_estatinas",
    "uso_antihipertensivos",
]

CATEGORICAL_FEATURES = [
    "tipo_alimentacion",
    "nivel_actividad",
    "antecedentes_familiares_tipo",
]

# Proporción del conjunto de test
TEST_SIZE = 0.2

# Número de particiones para validación cruzada
CV_FOLDS = 5
