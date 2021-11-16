import pickle

import numpy as np
import pandas as pd

try:
    from data_warehouse_utils.dataloader import DataLoader  # type: ignore  # noqa
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "`data_warehouse_utils` module not found. Have you obtained access to the DDW dataset?"
    ) from e

# get 21 days
T = 20


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

d_list = []


for i in range(len(effects_of_dex)):
    if i == 17:
        continue
    d = pd.read_csv("data/df_date_{}.csv".format(i), index_col=0)
    d_list.append(d)

df = pd.concat(d_list)
df["pacmed_name"][df["pacmed_name"] == "ph_unspecified"] = "ph_arterial"
df["pacmed_name"][df["pacmed_name"] == "lactate_unspecified"] = "lactate_arterial"
df["pacmed_name"][df["pacmed_name"] == "pco2_unspecified"] = "pco2_arterial"

df = df.groupby(["hash_patient_id", "pacmed_name", "date"]).agg("median").reset_index()

df_min = df.groupby("hash_patient_id").agg({"date": "min"}).reset_index()
df_min = df_min.rename(columns={"date": "date_min"})
df_min.head()
df_min.to_csv("data/date_admission.csv")
df = pd.merge(df, df_min, on=["hash_patient_id"])


df["date"] = pd.to_datetime(df["date"])
df["date_min"] = pd.to_datetime(df["date_min"])
df["days"] = (df.date - df.date_min).dt.days

df = df[df.days <= T]


iix_n = pd.MultiIndex.from_product([np.unique(df.days), np.unique(df.hash_patient_id)])

arr = (
    df.pivot_table("numerical_value", ["days", "hash_patient_id"], "pacmed_name", aggfunc="median")
    .reindex(iix_n)
    .to_numpy()
    .reshape(df.days.nunique(), df.hash_patient_id.nunique(), -1)
)

d_m = np.nanmean(arr, axis=(0, 1))
d_sd = np.nanstd(arr, axis=(0, 1))
arr_norm = (arr - d_m) / d_sd

arr_mask = np.isnan(arr_norm)
arr_norm[arr_mask] = 0
arr_mask = 1.0 - arr_mask

pickle.dump(arr_norm, open("data/array_xt.pkl", "wb"))
pickle.dump(arr_mask, open("data/array_xt_mask.pkl", "wb"))
pickle.dump(d_m, open("data/array_xt_mean.pkl", "wb"))
pickle.dump(d_sd, open("data/array_xt_std.pkl", "wb"))

# get static features

dl = DataLoader()


static_var = [
    "age",
    "gender",
    "bmi",
]
comor = [
    "cirrhosis",
    "chronic_dialysis",
    "chronic_renal_insufficiency",
    "diabetes",
    "cardiovascular_insufficiency",
    "copd",
    "respiratory_insufficiency",
    "immunodeficiency",
]

dfc = dl.get_comorbidities()
dfc = dfc[comor + ["hash_patient_id"]]

d_pat = pd.DataFrame(df.hash_patient_id.unique(), columns=["hash_patient_id"])

dfc = pd.merge(d_pat, dfc, on=["hash_patient_id"], how="left").reset_index()
del dfc["index"]

patients = dl.get_episodes()
patients = patients[static_var + ["hash_patient_id"]]
patients = patients.groupby(["hash_patient_id"]).agg("first").reset_index()

dfp = pd.merge(dfc, patients, on=["hash_patient_id"], how="left")

dfp.to_csv("data/static_covariates.csv")

dfp["gender"] = dfp["gender"] == "M"
dfp["age"] = (dfp["age"] - dfp["age"].mean()) / dfp["age"].std()
dfp["bmi"] = (dfp["bmi"] - dfp["bmi"].mean()) / dfp["bmi"].std()

dfp = dfp.fillna(dfp.median())

del dfp["hash_patient_id"]
pat_arr = dfp.values
pat_arr = pat_arr * 1.0
pat_arr = np.array(pat_arr, np.float64)

pickle.dump(pat_arr, open("data/array_x_constant.pkl", "wb"))
