# Cycling Behaviour in Flanders

## Objective

This project investigates what drives cycling volume in Flanders using automatic bicycle count data from AWV.

The analysis combines temporal cycling behaviour, spatial variation, weather conditions, behavioural segmentation, explainable machine learning, and interactive dashboarding.

## Data

Place your raw datasets here:

```text
data/raw/
```

Expected files:

```text
data-2019-08.csv
data-2019-09.csv
...
data-2026-05.csv
sites.csv
richtingen.csv
```

## Run Locally

```bash
pip install -r requirements.txt
python main.py
shiny run app.py
```

Open the dashboard at the URL shown by Shiny, usually:

```text
http://localhost:8000
```

## Run with Docker

```bash
docker build -t cycling-flanders-dashboard .
docker run -p 8000:8000 cycling-flanders-dashboard
```

## Course Alignment

This project demonstrates Python, OOP, pandas/numpy, scikit-learn pipelines, ML algorithms, Shiny dashboarding, Git-ready structure, Docker containerisation and scalable data processing.
