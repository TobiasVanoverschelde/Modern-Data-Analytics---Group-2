import pandas as pd

COLUMN_NAMES = ["site_id", "richting", "voertuig_type", "van", "tot", "aantal"]

SITES_COLS = ["site_id", "naam", "lon", "lat", "gemeente", "beheerder",
              "paalnummer", "code", "locatie", "interval_min", "installatie_datum"]

def load_counts(data_dir, pattern="data-*.csv"):
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")

    frames = [
        pd.read_csv(f, sep=",", header=None, names=COLUMN_NAMES, low_memory=False)
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} files | {len(df):,} rows")
    return df


def load_sites(data_dir):
    sites = pd.read_csv(
        data_dir / "sites.csv",
        sep=",",
        header=None,
        names=SITES_COLS,
        low_memory=False,
    )
    sites["site_id"] = sites["site_id"].astype(str)
    return sites


def load_directions(data_dir):
    return pd.read_csv(data_dir / "richtingen.csv")