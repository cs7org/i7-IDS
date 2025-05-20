from ids_expt.data.pfi import PermutationFeatureImportance
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

frac = 1.0  # 1.0
data_df = pd.read_csv(
    r"E:\MSc Works\IDS\data\Custom_DNP3_Parser\combined_45_timeout.csv"
)
data_df.columns = data_df.columns.str.strip()

df = data_df.sample(frac=frac, replace=False, random_state=42)
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=[np.number])
features = [
    f
    for f in X.columns.tolist()
    if not (f in ["frameSrc", "frameDst", "frameProtocol"] or "Unnamed" in f)
]
X = X[features]
pfi = PermutationFeatureImportance(
    model_object=RandomForestClassifier(random_state=42),
    X=X,
    y=df["Label"],
    split_size=0.2,
    random_state=42,
    k_fold=5,
    handle_imbalance=True,
    oversampling=False,
    classwise=True,
    out_dir=r"E:\MSc Works\IDS\local\pfi",
    permute_rate=0.95,
)
results = pfi.fit()
pfi.plot()
