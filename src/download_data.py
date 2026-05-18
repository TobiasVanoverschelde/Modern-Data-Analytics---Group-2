import os
import requests

BASE_URL = "https://opendata.apps.mow.vlaanderen.be/fietstellingen/"
RAW_DIR = "data/raw"

# Time range
START_YEAR, START_MONTH = 2019, 8
END_YEAR, END_MONTH = 2026, 4

def download(filename):
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path):
        return True
    r = requests.get(BASE_URL + filename, timeout=60)
    if r.status_code == 200 and "html" not in r.headers.get("Content-Type", ""):
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  OK   {filename} ({len(r.content) / 1024:,.0f} KB)")
        return True
    print(f"  SKIP {filename}")
    return False


def monthly_filenames():
    names = []
    year, month = START_YEAR, START_MONTH
    while (year, month) <= (END_YEAR, END_MONTH):
        names.append(f"data-{year}-{month:02d}.csv")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return names


os.makedirs(RAW_DIR, exist_ok=True)

# Download monthly data
counts = monthly_filenames()
print(f"Downloading {len(counts)} count files")
n_ok = sum(download(name) for name in counts)
print(f"-> {n_ok}/{len(counts)} count files\n")

# Download metadata
print(f"Downloading metadata files")
n_meta = sum(download(name) for name in ['richtingen.csv','sites.csv'])
print(f"-> Success")