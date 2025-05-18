import pandas as pd
from loguru import logger
from pathlib import Path


def oversample_class(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Oversample the specified class in the DataFrame.
    """
    logger.info(f"Before oversampling: {df[label].value_counts()}")
    label_counts = df[label].value_counts()
    max_count = label_counts.max()
    for lbl in label_counts.index:
        count = label_counts[lbl]
        if count < max_count:
            needed = max_count - count
            logger.info(f"Label {lbl} needs {needed} samples")
            oversample_df = df[df[label] == lbl].sample(needed, replace=True)
            df = pd.concat([df, oversample_df], ignore_index=True)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42, replace=False).reset_index(drop=True)
    logger.info(f"After oversampling: {df[label].value_counts()}")
    return df


def undersample_class(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Undersample the specified class in the DataFrame.
    """
    logger.info(f"Before undersampling: {df[label].value_counts()}")
    label_counts = df[label].value_counts()
    min_count = label_counts.min()
    for lbl in label_counts.index:
        count = label_counts[lbl]
        if count > min_count:
            needed = count - min_count
            logger.info(f"Label {lbl} needs to be reduced by {needed} samples")
            undersample_df = df[df[label] == lbl].sample(needed, replace=False)
            df = df.drop(undersample_df.index)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42, replace=False).reset_index(drop=True)
    logger.info(f"After undersampling: {df[label].value_counts()}")
    return df


def load_csv_files(all_files: list[Path]) -> pd.DataFrame:
    """
    Load all CSV files into a single DataFrame.
    """
    if len(all_files) == 0:
        logger.warning("No CSV files found.")
        return pd.DataFrame()
    dataframes = []
    for file in all_files:
        logger.info(f"Loading {file}")

        df = pd.read_csv(file, low_memory=False)
        if "Label" in df.columns:
            if "No Label" in df["Label"].unique():
                del df["Label"]
        df.columns = [c.strip() for c in df.columns]
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def get_dnp_cic_data(
    dnp3data_root: Path = Path(r"E:\MSc Works\IDS\data\Custom_DNP3_Parser"),
    cicflowdata_root: Path = Path(r"E:\MSc Works\IDS\data\CICFlowMeter"),
    samples: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = dnp3data_root
    logger.info(f"Data root: {data_root}")
    all_files = list(data_root.glob("*.csv"))
    logger.info(f"Found {len(all_files)} files in {data_root}")

    dnp3_df = load_csv_files(all_files)
    logger.info(f"Combined dnp3_df shape: {dnp3_df.shape}")
    dnp3_df.firstPacketDIR = dnp3_df["firstPacketDIR"].apply(
        lambda x: 1 if x == "MASTER" else 0
    )
    ignore_columns = [
        "File",
        "flow ID",
        "binary_label",
        "Timestamp",
        "source IP",
        "destination IP",
        "date",
        "Unnamed: 0",
        "Unnamed: 0.1",
    ]
    dnp3_df = dnp3_df[[c for c in dnp3_df.columns if c not in ignore_columns]]

    data_root = cicflowdata_root
    logger.info(f"Data root: {data_root}")
    all_files = list(data_root.glob("*.csv"))
    logger.info(f"Found {len(all_files)} files in {data_root}")
    cicflow_df = load_csv_files(all_files)
    logger.info(f"Combined cicflow_df shape: {cicflow_df.shape}")

    valid_cols = [
        "Src Port",
        "Dst Port",
        "Protocol",
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Fwd Pkt Len Max",
        "Fwd Pkt Len Min",
        "Fwd Pkt Len Mean",
        "Fwd Pkt Len Std",
        "Bwd Pkt Len Max",
        "Bwd Pkt Len Min",
        "Bwd Pkt Len Mean",
        "Bwd Pkt Len Std",
        "Flow Byts/s",
        "Flow Pkts/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Tot",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Tot",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "Fwd Header Len",
        "Bwd Header Len",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "Pkt Len Min",
        "Pkt Len Max",
        "Pkt Len Mean",
        "Pkt Len Std",
        "Pkt Len Var",
        "FIN Flag Cnt",
        "SYN Flag Cnt",
        "RST Flag Cnt",
        "PSH Flag Cnt",
        "ACK Flag Cnt",
        "URG Flag Cnt",
        "CWE Flag Count",
        "ECE Flag Cnt",
        "Down/Up Ratio",
        "Pkt Size Avg",
        "Fwd Seg Size Avg",
        "Bwd Seg Size Avg",
        "Fwd Byts/b Avg",
        "Fwd Pkts/b Avg",
        "Fwd Blk Rate Avg",
        "Bwd Byts/b Avg",
        "Bwd Pkts/b Avg",
        "Bwd Blk Rate Avg",
        "Subflow Fwd Pkts",
        "Subflow Fwd Byts",
        "Subflow Bwd Pkts",
        "Subflow Bwd Byts",
        "Init Fwd Win Byts",
        "Init Bwd Win Byts",
        "Fwd Act Data Pkts",
        "Fwd Seg Size Min",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
        "Label",
    ]

    # get columns with values smaller than 0 if the value is numerical else 0
    neg_cols = []
    for col in cicflow_df.columns:
        if cicflow_df[col].dtype == "float64" or cicflow_df[col].dtype == "int64":
            if (cicflow_df[col] < 0).any():
                neg_cols.append(col)

    # filter df with only neg cols and neg rows
    neg_df = cicflow_df[neg_cols]
    neg_df = neg_df[neg_df < 0]

    # get columns with values infinity
    inf_cols = []

    for col in cicflow_df.columns:
        if cicflow_df[col].dtype == "float64" or cicflow_df[col].dtype == "int64":
            if (cicflow_df[col] == float("inf")).any():
                inf_cols.append(col)

    numerical_df = cicflow_df[
        [c for c in valid_cols if c not in neg_cols and c not in inf_cols]
    ]
    if samples > 0:
        dnp3_df = dnp3_df.sample(samples, random_state=42)
        numerical_df = numerical_df.sample(samples, random_state=42)

    return dnp3_df, numerical_df


# clf_df = dnp3_df.query('Label!="NORMAL"').copy()
# det_df = dnp3_df.copy()
# det_df["Label"] = det_df["Label"].apply(
#     lambda x: "NORMAL" if x == "NORMAL" else "MALICIOUS"
# )
# logger.info(f"clf_df shape: {clf_df.shape}")
# logger.info(f"det_df shape: {det_df.shape}")

# clf_df = oversample_class(clf_df, "Label")
# det_df = oversample_class(det_df, "Label")
# # det_df = undersample_class(det_df, "Label")
