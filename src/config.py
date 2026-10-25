"""Paths and constants."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT/<class>/<class>-<n>/{img/, groundtruth.txt, ...}; override with LASOT_ROOT
DATA_ROOT = Path(os.environ.get("LASOT_ROOT", REPO_ROOT / "data"))

# default: every class folder in DATA_ROOT; override with LASOT_CLASSES=person,cat
CLASSES = (os.environ["LASOT_CLASSES"].split(",") if os.environ.get("LASOT_CLASSES") else
           sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name != "att") if DATA_ROOT.is_dir() else [])

SPLIT_FILE = DATA_ROOT / "testing_set.txt"

ATTRIBUTE_DIR = DATA_ROOT / "att"
# column order of the attribute files (att_fig_name in the LaSOT toolkit)
ATTRIBUTE_ORDER = ["IV", "POC", "DEF", "MB", "CM", "ROT", "BC", "VC", "SV", "FOC", "FM", "OV", "LR", "ARC"]

FPS = 30
RESULTS_DIR = REPO_ROOT / "results" / "eda"
FIG_DIR = RESULTS_DIR / "figures"

# thresholds of the derived attributes (LaSOT paper, Table 2)
SV_RANGE = (0.5, 2.0)
ARC_RANGE = (0.5, 2.0)
LR_AREA_PX = 1000
FM_DISP_RATIO = 1.0
