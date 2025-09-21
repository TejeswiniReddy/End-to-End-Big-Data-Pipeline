import pandas as pd
import matplotlib.pyplot as plt

pickup = pd.read_csv("eda_outputs/pickup_by_hour/part-00000-6375aa8d-d6c2-442a-89cf-ca1c2f07e2dc-c000.csv", header=None, names=["hour", "count"])
weekday = pd.read_csv("eda_outputs/weekday_counts/part-00000-a60631aa-6eef-4d90-9097-c17d29f26671-c000.csv", header=None, names=["weekday", "count"])
tips = pd.read_csv("eda_outputs/tip_percent/part-00000-682568fb-91ea-48f5-ba66-7d9e0cd31c5e-c000.csv", header=None, names=["payment_type", "avg_tip_pct"])

pickup.plot(kind='bar', x='hour', y='count', legend=False)
plt.title("Trip Counts by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Trips")
plt.tight_layout()
plt.grid(True)
plt.show()

weekday.plot(kind='bar', x='weekday', y='count', legend=False)
plt.title("Trips by Weekday")
plt.xlabel("Weekday (1=Sun)")
plt.ylabel("Trips")
plt.tight_layout()
plt.grid(True)
plt.show()

tips.plot(kind='bar', x='payment_type', y='avg_tip_pct', legend=False)
plt.title("Avg Tip % by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Tip %")
plt.tight_layout()
plt.grid(True)
plt.show()
