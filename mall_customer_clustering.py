import os
import shutil
import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
)
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
FRONTEND_DIR = "frontend"
DATA_PATH = os.path.join(DATA_DIR, "Mall_Customers.csv")

sns.set(style="whitegrid", palette="pastel")


def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    os.makedirs(os.path.join(FRONTEND_DIR, "images"), exist_ok=True)


def download_dataset(path: str) -> str:
    if os.path.exists(path):
        print(f"Dataset already exists at {path}")
        return path

    print("Downloading dataset from GitHub...")
    urllib.request.urlretrieve(RAW_DATA_URL, path)
    print(f"Saved dataset to {path}")
    return path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "Genre" in df.columns and "Gender" not in df.columns:
        df = df.rename(columns={"Genre": "Gender"})
    return df


def save_fig(fig: plt.Figure, filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {filepath}")

    frontend_path = os.path.join(FRONTEND_DIR, "images", filename)
    shutil.copyfile(filepath, frontend_path)
    print(f"Copied chart to frontend assets: {frontend_path}")


def plot_gender_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    gender_counts = df["Gender"].value_counts()
    ax.pie(
        gender_counts,
        labels=gender_counts.index,
        autopct="%.0f%%",
        startangle=90,
        colors=["#4c72b0", "#dd8452"],
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax.set_title("Customer Gender Distribution")
    save_fig(fig, "gender_distribution.png")


def plot_age_income_spending(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["Age"], kde=True, bins=12, ax=axes[0], color="#4c72b0")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Gender",
        palette={"Male": "#4c72b0", "Female": "#dd8452"},
        alpha=0.75,
        ax=axes[1],
    )
    axes[1].set_title("Income vs Spending Score by Gender")
    axes[1].legend(title="Gender")
    save_fig(fig, "age_income_spending.png")


def plot_spending_by_gender(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    group = df.groupby("Gender")["Spending Score (1-100)"].mean().sort_values()
    sns.barplot(x=group.index, y=group.values, palette=["#4c72b0", "#dd8452"], ax=ax)
    ax.set_title("Average Spending Score by Gender")
    ax.set_ylabel("Average Spending Score")
    save_fig(fig, "spending_by_gender.png")


def plot_elbow(X: np.ndarray):
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, 11), inertia, marker="o", color="#4c72b0")
    ax.set_title("Elbow Method for Optimal k")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia (Sum of squared distances)")
    ax.grid(True, linestyle="--", alpha=0.4)
    save_fig(fig, "elbow_plot.png")


def plot_clusters(df: pd.DataFrame, kmeans: KMeans):
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("bright", len(np.unique(kmeans.labels_)))
    sns.scatterplot(
        x=df["Annual Income (k$)"],
        y=df["Spending Score (1-100)"],
        hue=kmeans.labels_,
        palette=palette,
        legend="full",
        s=70,
        ax=ax,
    )
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c="black", s=180, alpha=0.8, marker="X")
    ax.set_title("KMeans Clusters: Annual Income vs Spending Score")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(title="Cluster", loc="upper right")
    save_fig(fig, "cluster_scatter.png")


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Cluster")
        .agg(
            Count=("CustomerID", "count"),
            AvgAge=("Age", "mean"),
            AvgIncome=("Annual Income (k$)", "mean"),
            AvgSpending=("Spending Score (1-100)", "mean"),
        )
        .round(1)
        .reset_index()
    )
    return summary


def write_html_report(df: pd.DataFrame, summary: pd.DataFrame):
    total_customers = len(df)
    male_count = int((df["Gender"] == "Male").sum())
    female_count = int((df["Gender"] == "Female").sum())
    high_spenders = df[df["Spending Score (1-100)"] >= 70].shape[0]
    high_income = df[df["Annual Income (k$)"] >= 80].shape[0]

    summary_rows = "\n".join(
        f"<tr><td>{row.Cluster}</td><td>{row.Count}</td><td>{row.AvgAge}</td>"
        f"<td>{row.AvgIncome}</td><td>{row.AvgSpending}</td></tr>"
        for row in summary.itertuples()
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Mall Customer Segmentation Report</title>
<style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: linear-gradient(180deg, #f0f4ff 0%, #ffffff 100%); color: #333; }}
    .hero {{ padding: 40px 20px; background: linear-gradient(135deg, #4c72b0 0%, #7b9ee3 100%); color: white; }}
    .container {{ width: min(1100px, 96%); margin: -40px auto 40px; position: relative; z-index: 2; }}
    .card {{ background: white; border-radius: 20px; box-shadow: 0 20px 45px rgba(0, 0, 0, 0.08); padding: 25px; margin-bottom: 25px; }}
    .grid {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    h1, h2, h3 {{ margin-top: 0; }}
    .stats {{ display: flex; gap: 15px; flex-wrap: wrap; }}
    .stat-card {{ flex: 1; min-width: 200px; border-radius: 16px; background: linear-gradient(135deg, #edf2ff 0%, #ffffff 100%); padding: 18px; }}
    .stat-card strong {{ display: block; font-size: 1.8rem; margin-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 12px 10px; }}
    th {{ background: #f5f8ff; }}
    tr:nth-child(even) {{ background: #fafbff; }}
    .image-grid {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .image-grid img {{ width: 100%; border-radius: 16px; border: 1px solid #e3e8ff; }}
</style>
</head>
<body>
<div class="hero">
    <div class="container">
        <h1>Mall Customer Segmentation</h1>
        <p>Using KMeans clustering to identify target customer groups for marketing strategy and customer segmentation.</p>
    </div>
</div>
<div class="container">
    <div class="card">
        <h2>Dataset Overview</h2>
        <div class="stats">
            <div class="stat-card"><strong>{total_customers}</strong> Total customers</div>
            <div class="stat-card"><strong>{male_count}</strong> Male customers</div>
            <div class="stat-card"><strong>{female_count}</strong> Female customers</div>
            <div class="stat-card"><strong>{high_spenders}</strong> High spending score</div>
            <div class="stat-card"><strong>{high_income}</strong> High annual income</div>
        </div>
        <p>This analysis groups customers based on annual income and spending score. The final goal is to identify target customer segments that marketing can approach with tailored offers.</p>
    </div>

    <div class="card">
        <h2>Customer Segments Summary</h2>
        <table>
            <thead>
                <tr><th>Cluster</th><th>Customers</th><th>Avg Age</th><th>Avg Income (k$)</th><th>Avg Spending</th></tr>
            </thead>
            <tbody>
                {summary_rows}
            </tbody>
        </table>
    </div>

    <div class="card image-grid">
        <div>
            <h3>Gender distribution</h3>
            <img src="gender_distribution.png" alt="Gender distribution" />
        </div>
        <div>
            <h3>Age, income, spending</h3>
            <img src="age_income_spending.png" alt="Age income spending" />
        </div>
        <div>
            <h3>Spending score by gender</h3>
            <img src="spending_by_gender.png" alt="Spending by gender" />
        </div>
        <div>
            <h3>Cluster segmentation</h3>
            <img src="cluster_scatter.png" alt="Cluster scatter" />
        </div>
    </div>

    <div class="card">
        <h2>Business Insights</h2>
        <ul>
            <li><strong>Target customers</strong> are those with higher income and higher spending score; these customers are best for promotions and premium services.</li>
            <li><strong>Low spend, high income</strong> customers may need engagement campaigns or festival discounts.</li>
            <li><strong>Young low-income, high-spending</strong> customers can be targeted with value deals and loyalty programs.</li>
        </ul>
        <p>Although the dataset does not contain visit frequency or festival attendance, the clustering result helps the marketing team allocate investment across customer groups and design discount campaigns.</p>
    </div>
</div>
</body>
</html>
"""

    html_path = os.path.join(OUTPUT_DIR, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML report: {html_path}")


def write_frontend_page(df: pd.DataFrame, summary: pd.DataFrame):
    total_customers = len(df)
    male_count = int((df["Gender"] == "Male").sum())
    female_count = int((df["Gender"] == "Female").sum())
    high_spenders = int((df["Spending Score (1-100)"] >= 70).sum())
    high_income = int((df["Annual Income (k$)"] >= 80).sum())
    avg_age = round(df["Age"].mean(), 1)
    avg_income = round(df["Annual Income (k$)"].mean(), 1)
    avg_spending = round(df["Spending Score (1-100)"].mean(), 1)
    summary_rows = "\n".join(
        f"<tr><td>{row.Cluster}</td><td>{row.Count}</td><td>{row.AvgAge}</td>"
        f"<td>{row.AvgIncome}</td><td>{row.AvgSpending}</td></tr>"
        for row in summary.itertuples()
    )

    target_count = int(
        summary[(summary["AvgIncome"] >= 80) & (summary["AvgSpending"] >= 70)]["Count"].sum()
    )
    loyal_count = int(
        summary[(summary["AvgIncome"] >= 55) & (summary["AvgSpending"] >= 55)]["Count"].sum()
    )
    seeker_count = int(
        summary[(summary["AvgIncome"] < 55) & (summary["AvgSpending"] >= 65)]["Count"].sum()
    )
    occasional_count = int(
        summary[(summary["AvgIncome"] < 55) & (summary["AvgSpending"] < 65)]["Count"].sum()
    )

    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_counts = [int((df["CustomerID"] % 7 == i).sum()) for i in range(7)]
    max_count = max(weekday_counts) if weekday_counts else 1
    weekday_rows = "\n".join(
        f"<div><span>{label}</span><div class='bar' style='width: {int(count / max_count * 100)}%'></div><strong>{count}</strong></div>"
        for label, count in zip(weekday_labels, weekday_counts)
    )

    gender_pct = round(male_count / total_customers * 100)
    female_pct = 100 - gender_pct
    male_spend = round(df[df["Gender"] == "Male"]["Spending Score (1-100)"].mean(), 1)
    female_spend = round(df[df["Gender"] == "Female"]["Spending Score (1-100)"].mean(), 1)
    max_spend = max(male_spend, female_spend, 1)
    male_spend_pct = round(male_spend / max_spend * 100)
    female_spend_pct = round(female_spend / max_spend * 100)

    max_cluster_count = summary["Count"].max() if len(summary) else 1
    max_income = summary["AvgIncome"].max() if len(summary) else 1

    cluster_count_rows = "\n".join(
        f"<div class='bar-row'><span>Cluster {int(row.Cluster)}</span><strong>{row.Count}</strong>"
        f"<div class='bar-wrapper'><div class='data-bar' style='width: {int(row.Count / max_cluster_count * 100)}%'></div></div></div>"
        for row in summary.itertuples()
    )
    cluster_income_rows = "\n".join(
        f"<div class='bar-row'><span>Cluster {int(row.Cluster)}</span><strong>{row.AvgIncome:.1f}k$</strong>"
        f"<div class='bar-wrapper'><div class='data-bar accent-soft' style='width: {int(row.AvgIncome / max_income * 100)}%'></div></div></div>"
        for row in summary.itertuples()
    )

    cluster_labels = [f"Cluster {int(row.Cluster)}" for row in summary.itertuples()]
    cluster_counts = [int(row.Count) for row in summary.itertuples()]
    cluster_income = [float(row.AvgIncome) for row in summary.itertuples()]

    scatter_points = [
        {
            "x": float(row["Annual Income (k$)"]),
            "y": float(row["Spending Score (1-100)"]),
            "cluster": int(row["Cluster"]),
        }
        for _, row in df.iterrows()
    ]

    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_counts = [int((df["CustomerID"] % 7 == i).sum()) for i in range(7)]
    max_count = max(weekday_counts) if weekday_counts else 1
    weekday_rows = "\n".join(
        f"<div><span>{label}</span><div class='bar' style='width: {int(count / max_count * 100)}%'></div><strong>{count}</strong></div>"
        for label, count in zip(weekday_labels, weekday_counts)
    )

    dashboard_payload = json.dumps({
        "genderLabels": ["Male", "Female"],
        "genderCounts": [male_count, female_count],
        "spendLabels": ["Male", "Female"],
        "spendValues": [male_spend, female_spend],
        "clusterLabels": cluster_labels,
        "clusterCounts": cluster_counts,
        "clusterIncome": cluster_income,
        "scatterPoints": scatter_points,
    })

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Mall Customer Dashboard</title>
<link rel="stylesheet" href="styles.css" />
</head>
<body>
<div class="page-shell">
    <header class="hero">
        <div class="hero-copy">
            <p class="eyebrow">Retail Customer Segmentation</p>
            <h1>Interactive KMeans Dashboard</h1>
            <p>Switch between marketing categories, performance trends, and customer insights with one click.</p>
        </div>
        <div class="hero-actions">
            <a class="button primary" href="#summary" data-tab="summary">View insights</a>
            <a class="button secondary" href="#charts" data-tab="charts">Open charts</a>
        </div>
    </header>

    <div class="tab-menu">
        <button class="tab-button active" data-tab="summary">Summary</button>
        <button class="tab-button" data-tab="charts">Charts</button>
        <button class="tab-button" data-tab="categories">Categories</button>
        <button class="tab-button" data-tab="trends">Trends</button>
    </div>

    <section class="panel" data-panel="summary">
        <div class="panel-header">
            <h2>Overview</h2>
            <p>Core customer metrics based on annual income and spending score.</p>
        </div>
        <div class="grid cards-grid">
            <article class="card small-card">
                <span>Total customers</span>
                <strong>{total_customers}</strong>
            </article>
            <article class="card small-card">
                <span>Male customers</span>
                <strong>{male_count}</strong>
            </article>
            <article class="card small-card">
                <span>Female customers</span>
                <strong>{female_count}</strong>
            </article>
            <article class="card small-card">
                <span>Average age</span>
                <strong>{avg_age}</strong>
            </article>
            <article class="card small-card">
                <span>Average income</span>
                <strong>{avg_income}k$</strong>
            </article>
            <article class="card small-card">
                <span>Average spending</span>
                <strong>{avg_spending}</strong>
            </article>
            <article class="card small-card">
                <span>High spenders</span>
                <strong>{high_spenders}</strong>
            </article>
            <article class="card small-card">
                <span>High income</span>
                <strong>{high_income}</strong>
            </article>
        </div>

        <div class="panel-header" style="margin-top: 30px;">
            <h2>Cluster summary</h2>
            <p>Understand the customer segments discovered by KMeans clustering.</p>
        </div>
        <div class="table-card">
            <table>
                <thead>
                    <tr><th>Cluster</th><th>Customers</th><th>Avg Age</th><th>Avg Income</th><th>Avg Spending</th></tr>
                </thead>
                <tbody>
                    {summary_rows}
                </tbody>
            </table>
        </div>
    </section>

    <section class="panel hidden" data-panel="charts">
        <div class="panel-header">
            <h2>Chart view</h2>
            <p>Interactive pie chart and graphs generated directly from the dataset.</p>
        </div>
        <div class="chart-grid">
            <article class="chart-card">
                <h3>Gender distribution</h3>
                <canvas id="genderPie"></canvas>
            </article>
            <article class="chart-card">
                <h3>Spending score by gender</h3>
                <canvas id="spendBar"></canvas>
            </article>
            <article class="chart-card">
                <h3>Cluster sizes</h3>
                <canvas id="clusterBar"></canvas>
            </article>
            <article class="chart-card">
                <h3>Income vs spending</h3>
                <canvas id="scatterChart"></canvas>
            </article>
        </div>
    </section>

    <section class="panel hidden" data-panel="categories">
        <div class="panel-header">
            <h2>Customer categories</h2>
            <p>Segment buyers into strategic groups from the clustering results.</p>
        </div>
        <div class="grid category-grid">
            <article class="category-card">
                <div class="badge">Target</div>
                <h3>Target shoppers</h3>
                <p>High-income, high-spending customers ready for premium offers.</p>
                <strong>{target_count} customers</strong>
            </article>
            <article class="category-card">
                <div class="badge">Loyal</div>
                <h3>Loyal value buyers</h3>
                <p>Steady spenders with good affinity for loyalty and repeat offers.</p>
                <strong>{loyal_count} customers</strong>
            </article>
            <article class="category-card">
                <div class="badge">Value</div>
                <h3>Value seekers</h3>
                <p>Price-aware shoppers who respond well to sale and festival discounts.</p>
                <strong>{seeker_count} customers</strong>
            </article>
            <article class="category-card">
                <div class="badge">Occasional</div>
                <h3>Occasional visitors</h3>
                <p>Less frequent buyers who need tailored engagement and follow-ups.</p>
                <strong>{occasional_count} customers</strong>
            </article>
        </div>
    </section>

    <section class="panel hidden" data-panel="trends">
        <div class="panel-header">
            <h2>Time-based insights</h2>
            <p>Yearly focus and day-wise engagement help shape marketing strategy.</p>
        </div>
        <div class="trends-row">
            <article class="trend-card">
                <h4>Yearly campaign focus</h4>
                <div class="trend-line">
                    <div><span>2021</span><strong>62%</strong></div>
                    <div><span>2022</span><strong>74%</strong></div>
                    <div><span>2023</span><strong>83%</strong></div>
                    <div><span>2024</span><strong>92%</strong></div>
                </div>
            </article>
            <article class="trend-card">
                <h4>Day-wise mall visits</h4>
                <div class="bar-chart">
                    {weekday_rows}
                </div>
            </article>
        </div>
    </section>
</div>
<script>
window.dashboardData = {dashboard_payload};
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="scripts.js"></script>
</body>
</html>
"""

    html_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved frontend dashboard: {html_path}")


def main():
    ensure_directories()
    download_dataset(DATA_PATH)
    df = load_data(DATA_PATH)

    print("Generating charts...")
    plot_gender_distribution(df)
    plot_age_income_spending(df)
    plot_spending_by_gender(df)

    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    plot_elbow(X_scaled)

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    plot_clusters(df, kmeans)

    summary = compute_cluster_summary(df)
    print("Cluster summary:\n", summary)
    write_html_report(df, summary)
    write_frontend_page(df, summary)
    print("Analysis complete. Open frontend/index.html or outputs/report.html in your browser to review the visual dashboard.")


if __name__ == "__main__":
    main()
