from pathlib import Path
from collections import defaultdict

output_root = Path(r"E:\MSc Works\IDS\notebooks\output")

lbl_cnts = defaultdict(int)

for img in output_root.rglob("*.png"):
    if "hot.png" in img.name:
        continue
    lbl = img.stem.split("_")[2:]
    lbl = "_".join(lbl)
    lbl_cnts[lbl] += 1
print("Label counts:")
for lbl, cnt in lbl_cnts.items():
    print(f"{lbl}: {cnt}")
