import shutil
from pathlib import Path

import pandas as pd

prob = Path("outputs/reports/v2_download_problems.txt").read_text().strip().splitlines()
miss = [l.split("\t")[0] for l in prob if "404" in l]
other = [l for l in prob if "404" not in l]
print("404 files:", len(miss), "| other failures:", len(other))
assert not other, "non-404 failures present; do not exclude, investigate first"

img = pd.read_csv("manifests/v2_candidate_images.csv", dtype={"case_id": str})
shutil.copy("manifests/v2_candidate_images.csv", "manifests/v2_candidate_images_before_exclusion.csv")
hit = img[img.image_path.isin(miss)]
print(hit[["case_id", "label", "image_idx", "image_path"]].to_string(index=False))

keep = img[~img.image_path.isin(miss)]
lost = set(hit.case_id) - set(keep.case_id)
print("cases left with NO image (dropped):", len(lost))
keep.to_csv("manifests/v2_candidate_images.csv", index=False)

cases = pd.read_csv("manifests/v2_cases.csv", dtype={"case_id": str})
cases = cases[~cases.case_id.isin(lost)]
cases.to_csv("manifests/v2_cases.csv", index=False)
print("image rows:", len(keep), "| cases:", cases.case_id.nunique())