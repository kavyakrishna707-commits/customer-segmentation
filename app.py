
from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "customers.csv"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Load the customer dataset
df = pd.read_csv(DATA_FILE)

# Select the features used for clustering
features = ["AnnualIncome", "SpendingScore"]
X = df[features]

# Scale the features so they are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Cluster summaries for the dashboard
summary = (
    df.groupby("Cluster")
    .agg(
        Customers=("CustomerID", "count"),
        AverageIncome=("AnnualIncome", "mean"),
        AverageSpending=("SpendingScore", "mean")
    )
    .round(2)
    .reset_index()
)

# Create the cluster visualization
plt.figure(figsize=(8, 5))

for cluster in sorted(df["Cluster"].unique()):
    group = df[df["Cluster"] == cluster]
    plt.scatter(
        group["AnnualIncome"],
        group["SpendingScore"],
        label=f"Segment {cluster + 1}",
        s=80
    )

# Convert scaled cluster centres back to original units
centres = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centres[:, 0],
    centres[:, 1],
    marker="X",
    s=220,
    label="Centroids"
)

plt.xlabel("Annual Income (thousands)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Using K-Means")
plt.legend()
plt.tight_layout()
plt.savefig(STATIC_DIR / "cluster_plot.png", dpi=120)
plt.close()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            income = float(request.form["income"])
            spending = float(request.form["spending"])

            if income < 0 or not 1 <= spending <= 100:
                raise ValueError(
                    "Income must be non-negative and spending score must be 1–100."
                )

            customer = pd.DataFrame(
                [[income, spending]],
                columns=features
            )

            customer_scaled = scaler.transform(customer)
            cluster = int(kmeans.predict(customer_scaled)[0])

            prediction = {
                "segment": cluster + 1,
                "income": income,
                "spending": spending
            }

        except ValueError as e:
            error = str(e) or "Please enter valid numbers."

    return render_template(
        "index.html",
        summary=summary.to_dict(orient="records"),
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
