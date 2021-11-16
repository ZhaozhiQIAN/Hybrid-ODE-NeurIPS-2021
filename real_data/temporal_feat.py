import pandas as pd

try:
    from data_warehouse_utils.dataloader import DataLoader  # type: ignore  # noqa
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "`data_warehouse_utils` module not found. Have you obtained access to the DDW dataset?"
    ) from e

# get patient cohort 3 <= dt < 31
dl = DataLoader()

patients = dl.get_admissions()
patients["los"] = patients.discharge_timestamp - patients.admission_timestamp

lo = pd.to_timedelta(3, unit="D")
hi = pd.to_timedelta(31, unit="D")

patients_filtered = patients[(patients["los"] >= lo) & (patients["los"] < hi)]
patient_ids = patients_filtered.hash_patient_id

effects_of_dex = [
    "pao2_over_fio2",
    "peep",
    "adjusted_sofa_total_partial",
    "temperature",
    "arterial_blood_pressure_mean",
    "heart_rate",
    "bilirubin_total",
    "thrombocytes",
    "leukocytes",
    "creatinine",
    "c_reactive_protein",
    "lactate_arterial",
    "lactate_unspecified",
    "creatine_kinase",
    "glucose",
    "alanine_transaminase",
    "aspartate_transaminase",
    "position",  # Categorical  variable; only prone and supine are of importance
    "tidal_volume",
    "driving_pressure",
    "fio2",
    "lung_compliance_static",
    "respiratory_rate_measured_ventilator",
    "pressure_above_peep",
    "pco2_arterial",
    "pco2_unspecified",
    "ph_arterial",
    "ph_unspecified",
]


def save_df(i):
    xt = dl.get_single_timestamp(parameters=[effects_of_dex[i]], patients=list(patient_ids))
    xt["date"] = xt["effective_timestamp"].dt.date
    xt["hour"] = xt["effective_timestamp"].dt.hour

    df_hour = (
        xt.groupby(["hash_patient_id", "pacmed_name", "date", "hour"]).agg({"numerical_value": "median"}).reset_index()
    )
    df_date = xt.groupby(["hash_patient_id", "pacmed_name", "date"]).agg({"numerical_value": "median"}).reset_index()

    df_hour.to_csv("data/df_hour_{}.csv".format(i))
    df_date.to_csv("data/df_date_{}.csv".format(i))


for i in range(len(effects_of_dex)):

    try:
        save_df(i)
    except Exception:
        print(i, effects_of_dex[i])
