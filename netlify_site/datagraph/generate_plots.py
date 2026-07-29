import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Create images dir
os.makedirs('images', exist_ok=True)

# Generate mock data for 3.2
np.random.seed(42)
normal_data = np.random.normal(12.5, 2.0, 1000)
outliers = np.concatenate([np.random.uniform(0, 3, 20), np.random.uniform(22, 30, 30)])
data = np.concatenate([normal_data, outliers])
df = pd.DataFrame({'입도': data})

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.violinplot(x=df['입도'], ax=axes[0], color='lightcoral')
axes[0].set_title('이상치 제거 전 (Before)')

Q1 = df['입도'].quantile(0.25)
Q3 = df['입도'].quantile(0.75)
IQR = Q3 - Q1
clean_d = df[(df['입도'] >= Q1 - 1.5 * IQR) & (df['입도'] <= Q3 + 1.5 * IQR)]
sns.violinplot(x=clean_d['입도'], ax=axes[1], color='lightblue')
axes[1].set_title('이상치 제거 후 (After)')
plt.tight_layout()
plt.savefig('images/outlier_comparison.png', dpi=150)
plt.close()

# Generate mock data for 3.3 (Time Series missing values)
time = np.arange(100)
true_values = np.sin(time / 5.0) * 10 + 50
df2 = pd.DataFrame({'수분율': true_values})
# Drop some chunks of values to make missing gaps obvious
df2.loc[10:15, '수분율'] = np.nan
df2.loc[40:50, '수분율'] = np.nan
df2.loc[75:80, '수분율'] = np.nan

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(df2['수분율'], marker='o', linestyle='-', color='gray')
axes[0].set_title('보간 전 수분율 (끊어짐 발생)')

df2['수분율_보간'] = df2['수분율'].interpolate(method='linear')
axes[1].plot(df2['수분율_보간'], marker='o', linestyle='-', color='orange')
axes[1].set_title('선형 보간 후 수분율 (자연스럽게 연결됨)')
plt.tight_layout()
plt.savefig('images/interpolation_comparison.png', dpi=150)
plt.close()

print("Images generated successfully.")
