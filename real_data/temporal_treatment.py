import pickle

import numpy as np
import pandas as pd

try:
    from data_warehouse_utils.dataloader import DataLoader  # type: ignore  # noqa
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "`data_warehouse_utils` module not found. Have you obtained access to the DDW dataset?"
    ) from e

T = 20

# process raw data
dl = DataLoader()
DEXAMETHASONE = [
    "med_dexamethasone",
    "med_dexamethasone_and_antibiotics",
    "med_dexamethasone_and_antiinfectives",
    "med_dexamethasone_combinations",
]

pats = dl.get_patients()
meds = dl.get_medications(parameters=DEXAMETHASONE)
medications = meds.copy()

medications = medications[medications["pacmed_name"] == "med_dexamethasone"]
medications.total_dose = medications.total_dose.round(decimals=2)
medications.administration_route = medications.administration_route.fillna("intraveneus")
medications = medications[medications.administration_route.isin(["intraveneus", "INTRAVENEUS"])]

# joining
adm = pd.read_csv("data/date_admission.csv", index_col=0)
medications = medications[["hash_patient_id", "start_timestamp", "total_dose"]]
df_joined = pd.merge(adm, medications, how="left", on=["hash_patient_id"])
df_joined["time"] = (df_joined["start_timestamp"] - pd.to_datetime(df_joined["date_min"])).dt.days
df_joined = df_joined[df_joined["time"] <= T]
df_mat = df_joined.pivot_table("total_dose", ["hash_patient_id"], "time", aggfunc="sum").reset_index()
df_mat = pd.merge(adm, df_mat, how="left", on=["hash_patient_id"])
df_mat.to_csv("data/treatment.csv")
del df_mat["hash_patient_id"]
del df_mat["date_min"]
a_mat = df_mat.values
a_mat.shape
a_mat[np.isnan(a_mat)] = 0.0
a_mat = a_mat / a_mat.std()
a_mat = a_mat.T[:, :, None]
pickle.dump(a_mat, open("data/array_at.pkl", "wb"))
