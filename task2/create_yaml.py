from pathlib import Path
import yaml

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "trafic_data"
OUTPUT_YAML = TASK_DIR / "vehicle.yaml"

config = {
    "path": str(DATA_DIR),
    "train": "train/images",
    "val": "valid/images",
    "nc": 21,
    "names": [
        "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus",
        "car", "garbagevan", "human hauler", "minibus", "minivan",
        "motorbike", "pickup", "policecar", "rickshaw", "scooter",
        "suv", "taxi", "three wheelers -CNG-", "truck", "van", "wheelbarrow",
    ],
}

with open(OUTPUT_YAML, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"vehicle.yaml created successfully: {OUTPUT_YAML}")
