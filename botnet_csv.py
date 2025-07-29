import pandas as pd
import numpy as np

np.random.seed(42)

rows = []
target_distribution = {0: 0, 1: 0}

while len(rows) < 5000:
    login_Attempts = np.random.randint(0, 11)
    failed_Validations = np.random.randint(0, 13)
    avg_flow = np.random.choice([500, 530, 550])
    flow_noise = np.random.randint(-300, 1000)
    current_flow = avg_flow + flow_noise
    packets_flow = np.random.randint(1, 21)

    # Rule-based Labeling
    if login_Attempts > 5 and failed_Validations > 5 and current_flow > 700:
        botnet = 1
    elif login_Attempts < 3 and abs(current_flow - avg_flow) < 20:
        botnet = 0
    else:
        botnet = np.random.choice([0, 1], p=[0.5, 0.5])

    # Balance the dataset dynamically
    if target_distribution[botnet] < 2500:
        rows.append([
            login_Attempts,
            failed_Validations,
            avg_flow,
            current_flow,
            packets_flow,
            botnet
        ])
        target_distribution[botnet] += 1

# Create DataFrame
columns = ["login_Attempts", "failed_Validations", "avg_flow", "current_flow", "packets_flow", "botnet"]
df = pd.DataFrame(rows, columns=columns)

# Save CSV
df.to_csv("synthetic_botnet_dataset_large.csv", index=False)

print("✅ Dataset created with 5000 rows (Balanced classes)")
print(df['botnet'].value_counts())
