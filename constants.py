from pathlib import Path
import yaml


#loc_configs = Path.cwd() / "configs" / "var_config_volfactor.yml" #location of config file

with open(Path(__file__).resolve().parent / "configs" / "var_config_volfactor.yml") as f:
    CONFIGS = yaml.safe_load(f)

F_DATA = Path(__file__).resolve().parent / "data"/ "pricing_data_combined.csv" #filepath of data file



