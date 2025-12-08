import subprocess
from pathlib import Path
import json
from multiprocessing import Pool
from loguru import logger

map_file = Path(r"E:\MSc Works\IDS\notebooks\cic_file_mapping.json")
pcap_root = Path(r"E:\MSc Works\IDS\data\Total PCAP Files")
csv_root = Path(r"E:\MSc Works\IDS\data\CICFlowMeter")
output_dir = Path(r"E:\MSc Works\IDS\output_cic_labelled_sessions")
completed_files = [f.name for f in output_dir.iterdir()]
pcap_files = list(pcap_root.rglob("*.pcap"))

with open(map_file, "r") as f:
    file_mapping = json.load(f)

files = [f for f in csv_root.glob("*.csv") if f.is_file()]
files_sorted_by_size = sorted(files, key=lambda f: f.stat().st_size)
logger.info(
    f"Out of {len(files_sorted_by_size)} files, {len(completed_files)} are already processed."
)
cfnames = [
    f.stem.split(".")[0] for f in pcap_files if f.stem.split(".")[0] in completed_files
]
completed_files = [c for c in completed_files if c in cfnames]
logger.info(f"Processing {len(files_sorted_by_size) - len(completed_files)} files.")


def run_shell_script(args):
    csv_file, pcap_file, output_dir = args
    cmd = [
        r"C:\Program Files\Git\bin\bash.exe",
        str(Path(r"E:\MSc Works\IDS\feature_importance\pcap_label.sh").resolve()),
        str(csv_file),
        str(pcap_file),
    ]

    if output_dir:
        cmd.append(str(output_dir))
    try:
        # This will block until the script finishes, ensuring sequential execution
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully processed {csv_file.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {csv_file.name}: {e}")


if __name__ == "__main__":
    # Prepare arguments for multiprocessing
    num_workers = 5
    tasks = []
    for idx, cfile in enumerate(files_sorted_by_size):
        fname = cfile.name
        if fname not in file_mapping:
            print(f"Skipping {fname} as no matching pcap file found.")
            continue

        pcap_file = file_mapping[fname]
        pcap_path = pcap_root / pcap_file
        if pcap_path.stem.split(".")[0] in completed_files:
            logger.info(f"Skipping {fname} as it is already processed.")
            continue
        logger.info(f"Queueing {idx+1}/{len(files_sorted_by_size)}: {fname}")
        tasks.append((cfile, pcap_path, output_dir))

    # Use multiprocessing to process files in parallel

    with Pool(processes=num_workers) as pool:
        pool.map(run_shell_script, tasks)
