# Mall Customer Segmentation Project

This project demonstrates customer segmentation for a retail mall using KMeans clustering. It loads the Mall Customer dataset, performs exploratory data analysis, runs clustering, and generates visual charts plus an HTML report.

## Files

- `mall_customer_clustering.py`: Main analysis script. Downloads the dataset, trains KMeans, saves charts, and writes an HTML report.
- `requirements.txt`: Python dependencies.
- `data/Mall_Customers.csv`: Dataset file (downloaded automatically when the script runs).
- `outputs/`: Output directory for charts and report.

## Run instructions

1. Create a Python environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the clustering script:

```bash
python mall_customer_clustering.py
```

4. Open the friendly dashboard:

```bash
start frontend/index.html
```

5. Or open the report:

```bash
start outputs/report.html
```

## What this project includes

- Gender and spending visualizations
- Income vs spending scatter plots
- Elbow method to choose cluster count
- KMeans segmentation on income and spending score
- Summary tables and business-focused insights
- HTML report with descriptive cards and charts
