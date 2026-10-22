"""Central configuration for the LaSOT EDA. Edit here, not in the notebook."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Expected layout: DATA_ROOT/<class>/<class>-<n>/{img/, groundtruth.txt, ...}
# Override without editing the file:  export LASOT_ROOT=/path/to/lasot
DATA_ROOT = Path(os.environ.get("LASOT_ROOT", REPO_ROOT / "data"))

# Classes to analyse. Default: every class folder found in DATA_ROOT (fill it with
# scripts/fetch_lasot_annotations.py). Override: export LASOT_CLASSES=person,cat
CLASSES = (os.environ["LASOT_CLASSES"].split(",") if os.environ.get("LASOT_CLASSES") else
           sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name != "att") if DATA_ROOT.is_dir() else [])

# Official LaSOT test list (one sequence name per line, e.g. "person-3").
# Put it next to the data; set to None if you don't have it yet.
SPLIT_FILE = DATA_ROOT / "testing_set.txt"

# Official per-sequence attribute labels: ATTRIBUTE_DIR/<seq>.txt, 14 values of 1/0 (from the toolkit's annos/att).
ATTRIBUTE_DIR = DATA_ROOT / "att"
# Names of the 14 columns IN FILE ORDER, from att_fig_name in the toolkit's
# run_tracker_performance_evaluation.m. Still confirm with stats.check_attribute_columns() (FOC, OV ~ 1.0).
ATTRIBUTE_ORDER = ["IV", "POC", "DEF", "MB", "CM", "ROT", "BC", "VC", "SV", "FOC", "FM", "OV", "LR", "ARC"]

FPS = 30
RESULTS_DIR = REPO_ROOT / "results" / "eda"
FIG_DIR = RESULTS_DIR / "figures"

# Thresholds used for the *derived* (proxy) attributes, following paper Table 2.
SV_RANGE = (0.5, 2.0)    # box-area ratio vs first frame outside this -> SV
ARC_RANGE = (0.5, 2.0)   # aspect-ratio ratio vs first frame outside this -> ARC
LR_AREA_PX = 1000        # box area < this in at least one frame -> LR
FM_DISP_RATIO = 1.0      # centre displacement / sqrt(box area) between frames -> FM
