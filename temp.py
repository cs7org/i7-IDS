import pandas as pd
from pathlib import Path
from loguru import logger


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


data_root = Path(r"E:\MSc Works\IDS\data\Custom_DNP3_Parser")
logger.info(f"Data root: {data_root}")
all_files = list(data_root.glob("*.csv"))
logger.info(f"Found {len(all_files)} files in {data_root}")
combined_df = load_csv_files(all_files)
logger.info(f"Combined DataFrame shape: {combined_df.shape}")
