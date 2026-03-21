from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import pandas as pd
from ids_expt.core.defs import TOP_CIC_FEATURES
from pathlib import Path
from loguru import logger

from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
import numpy as np

out_dir = Path(r"E:\MSc Works\IDS\results") / "cic_ctgan_synthetic_data"
# df = pd.read_csv(r"E:\MSc Works\IDS\data\Custom_DNP3_Parser\combined_45_timeout.csv")
df = pd.read_csv(
    r"E:\MSc Works\IDS\data\cicflow_combined.csv",
    low_memory=False,
)
df.columns = df.columns.str.strip()
labels = df["Label"].unique().tolist()
features = TOP_CIC_FEATURES  # TOP_FEATURES
data = df[features + ["Label"]].copy()

if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)

max_lbl_cnt = df["Label"].value_counts().max()
logger.info(f"Lable counts: {df['Label'].value_counts().to_dict()}")
# CTGAN
combined_df = data.copy()
combined_df["is_synthetic"] = False
for label in labels:
    logger.info(f"Processing label: {label}")
    if label == "NORMAL":
        continue
    csv_dir = out_dir / f"ctgan_synthetic_data_{label}.csv"
    if csv_dir.exists():
        logger.info(f"File {csv_dir} already exists, skipping...")
        continue
    data = df.query(f"Label == '{label}'").sample(frac=1)

    data = data[~np.isinf(data.select_dtypes(include=[np.number])).any(axis=1)]
    X_train = data.drop(columns=["Label"]).copy()
    X_train = X_train[features].copy()
    y_train = data["Label"]
    counts = len(data)
    needed_samples = max_lbl_cnt - counts
    # Define metadata for the dataset
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_train)
    if (out_dir / f"ctgan_metadata_{label}.json").exists():
        # delete it
        (out_dir / f"ctgan_metadata_{label}.json").unlink()
    metadata.save_to_json(filepath=out_dir / f"ctgan_metadata_{label}.json")

    # CTGAN
    ctgan = CTGANSynthesizer(
        metadata,
        verbose=True,
        epochs=100,
        embedding_dim=32,
        generator_dim=(64, 64),
        discriminator_dim=(64, 64),
        batch_size=30,
    )
    ctgan.fit(X_train)

    # save model
    ctgan.save(out_dir / f"ctgan_model_{label}.pkl")
    # metadata.save(out_dir / f"ctgan_metadata_{label}.json")

    synthetic_data = ctgan.sample(needed_samples)

    diagnostic = run_diagnostic(
        real_data=X_train, synthetic_data=synthetic_data, metadata=metadata
    )
    quality_report = evaluate_quality(X_train, synthetic_data, metadata)
    synthetic_data["Label"] = label
    synthetic_data["is_synthetic"] = True
    synthetic_data["quality_score"] = quality_report.get_score()
    synthetic_data["diagnostic_score"] = diagnostic.get_score()
    synthetic_data.to_csv(csv_dir, index=False)
    # combined_df = pd.concat([combined_df, synthetic_data], ignore_index=True)
    # break
# combined_df.to_csv(out_dir / "ctgan_synthetic_data.csv", index=False)
