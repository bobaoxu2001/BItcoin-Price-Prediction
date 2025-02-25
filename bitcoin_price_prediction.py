# -*- coding: utf-8 -*-
"""Bitcoin Price Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1J0OAQSq5nw-sTN-slHEeVRSSwKayUbn2
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 设置随机种子确保结果可重复
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 加载数据
data_path = "filtered_df.csv"  # 替换为你的数据路径
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])  # 确保日期解析正确
print(f"数据形状: {df.shape}")

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_column, seq_length=100, feature_scaler=None, target_scaler=None):
        self.seq_length = seq_length
        self.features = data.drop(['date', target_column], axis=1)
        self.target = data[target_column].values.reshape(-1, 1)

        # 标准化特征和目标值
        self.feature_scaler = feature_scaler or StandardScaler()
        self.target_scaler = target_scaler or StandardScaler()
        self.scaled_features = self.feature_scaler.fit_transform(self.features)
        self.scaled_target = self.target_scaler.fit_transform(self.target)

        self.scaled_features = torch.FloatTensor(self.scaled_features)
        self.scaled_target = torch.FloatTensor(self.scaled_target)

    def __len__(self):
        return len(self.scaled_features) - self.seq_length

    def __getitem__(self, idx):
        x = self.scaled_features[idx:idx + self.seq_length]
        y = self.scaled_target[idx + self.seq_length]
        return x, y

class iTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, n_heads=4, n_layers=2, ff_dim=256, dropout=0.1):
        super(iTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x[-1, :, :]
        x = self.fc(x)
        return x

# 初始化参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_size = 3000
seq_length = 100
window_start = 62809
forecast_horizon = 100
step_size = 100
batch_size = 32

# 创建标准化器
feature_scaler = StandardScaler().fit(df.drop(['date', 'target_nexthour'], axis=1))
target_scaler = StandardScaler().fit(df['target_nexthour'].values.reshape(-1, 1))

# 初始化模型
input_dim = df.drop(['date', 'target_nexthour'], axis=1).shape[1]
model = iTransformer(input_dim=input_dim).to(device)

# 存储所有结果
all_predictions = []
all_actuals = []
all_dates = []

for i in range(0, len(df) - window_start - forecast_horizon + 1, step_size):
    current_start = window_start + i
    train_df = df.iloc[current_start-train_size:current_start].copy()
    test_df = df.iloc[current_start:current_start+forecast_horizon].copy()

    # 创建数据集
    train_dataset = TimeSeriesDataset(
        train_df, target_column='target_nexthour',
        seq_length=seq_length, feature_scaler=feature_scaler, target_scaler=target_scaler
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # 滑动窗口预测
    initial_sequence = train_df.iloc[-seq_length:].copy()
    predictions = []
    with torch.no_grad():
        for _ in range(forecast_horizon):
            features = initial_sequence.drop(['date', 'target_nexthour'], axis=1).values[-100:, :]
            scaled_features = feature_scaler.transform(features)
            x = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)
            scaled_pred = model(x).cpu().numpy()[0]
            pred = target_scaler.inverse_transform(scaled_pred.reshape(-1, 1))[0, 0]
            predictions.append(pred)
            next_row = initial_sequence.iloc[-1].copy()
            next_row['target_nexthour'] = pred
            initial_sequence = pd.concat([initial_sequence, pd.DataFrame([next_row])])

    # 存储结果
    actuals = test_df['target_nexthour'].values
    dates = test_df['date'].values
    all_predictions.extend(predictions)
    all_actuals.extend(actuals)
    all_dates.extend(dates)



# 保存结果
results_df = pd.DataFrame({
    'date': all_dates,
    'actual': all_actuals,
    'predicted': all_predictions
})
results_df.to_csv("predictions_results.csv", index=False)
print("结果已保存到 predictions_results.csv")

# 计算误差分布
errors = np.array(all_predictions) - np.array(all_actuals)

# 绘制误差分布
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, color='blue', alpha=0.7)
plt.title("Prediction Error Distribution", fontsize=16)
plt.xlabel("Error", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.grid(alpha=0.4)
plt.savefig("error_distribution.png")
plt.show()

# 计算每日收益
results_df['actual_return'] = results_df['actual'].pct_change()
results_df['predicted_return'] = results_df['predicted'].pct_change()

# 绘制每日收益对比
plt.figure(figsize=(12, 6))
plt.plot(results_df['date'], results_df['predicted_return'], label="Predicted Returns", marker='o', markersize=4)
plt.plot(results_df['date'], results_df['actual_return'], label="Actual Returns", linestyle='--', marker='x', markersize=4)
plt.title("Daily Predicted and Actual Returns", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Returns", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.savefig("daily_returns.png")
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate MSE and MAE
mse = mean_squared_error(all_actuals, all_predictions)
mae = mean_absolute_error(all_actuals, all_predictions)

# Print the results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Save the predictions and actuals to a CSV (if not already saved)
results_df = pd.DataFrame({
    'date': all_dates,
    'actual_price': all_actuals,
    'predicted_price': all_predictions
})
results_csv_path = "predictions_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")

def load_and_process_predictions(file_path):
    """Load and process prediction results."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Calculate returns
    df['Actual_Returns'] = df['actual_price'].pct_change()
    df['Predicted_Returns'] = df['predicted_price'].pct_change()

    # Rename columns for consistency
    df = df.rename(columns={
        'actual_price': 'Actual_Price',
        'predicted_price': 'Predicted_Price'
    })

    return df

# Load the processed predictions
predictions_df = load_and_process_predictions(results_csv_path)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_enhanced_predictions(predictions_df, save_path='enhanced_visualization'):
    """Create enhanced visualizations for predictions."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Full Timeline Prediction',
            'Last 7 Days',
            'Last 30 Days',
            'Returns Comparison',
            'Error Distribution',
            'Error Timeline'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]] * 3,
        vertical_spacing=0.1,
        horizontal_spacing=0.12
    )

    # Define colors and line styles
    colors = {'Actual': '#1f77b4', 'Predicted': '#ff7f0e'}
    line_styles = {
        'Actual': dict(color=colors['Actual'], width=2),
        'Predicted': dict(color=colors['Predicted'], width=2)
    }

    # Define timeframes
    timeframes = {'Full Period': None, 'Last 7 Days': 7, 'Last 30 Days': 30}
    row, col = 1, 1

    for timeframe_name, days in timeframes.items():
        # Subset data
        if days:
            df_window = predictions_df.iloc[-days:]
        else:
            df_window = predictions_df

        # Add Actual and Predicted Prices
        fig.add_trace(
            go.Scatter(
                x=df_window.index, y=df_window['Actual_Price'],
                name='Actual' if row == 1 and col == 1 else None,
                line=line_styles['Actual'],
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=df_window.index, y=df_window['Predicted_Price'],
                name='Predicted' if row == 1 and col == 1 else None,
                line=line_styles['Predicted'],
                showlegend=(row == 1 and col == 1)
            ),
            row=row, col=col
        )

        # Move to the next subplot
        col += 1
        if col > 2:
            col = 1
            row += 1

    # Returns Comparison (Last 30 Days)
    returns_window = predictions_df.iloc[-30:]
    fig.add_trace(
        go.Scatter(
            x=returns_window.index, y=returns_window['Actual_Returns'],
            name='Actual Returns',
            line=dict(color=colors['Actual'], width=1),
            showlegend=True
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=returns_window.index, y=returns_window['Predicted_Returns'],
            name='Predicted Returns',
            line=dict(color=colors['Predicted'], width=1),
            showlegend=True
        ),
        row=2, col=2
    )

    # Error Distribution
    error_pct = ((predictions_df['Actual_Price'] - predictions_df['Predicted_Price']) /
                 predictions_df['Actual_Price']) * 100
    fig.add_trace(
        go.Histogram(
            x=error_pct,
            name='Prediction Error',
            opacity=0.7,
            nbinsx=50,
            marker_color=colors['Predicted']
        ),
        row=3, col=1
    )

    # Error Timeline
    fig.add_trace(
        go.Scatter(
            x=predictions_df.index, y=error_pct,
            name='Prediction Error Timeline',
            line=dict(color=colors['Predicted'], width=1),
            showlegend=True
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        height=1500, width=1800,
        title_text='Comprehensive Bitcoin Price Prediction Analysis',
        showlegend=True
    )

    # Save and show
    fig.write_html(f"{save_path}.html")
    fig.show()

    return fig

# Plot enhanced visualizations
plot_enhanced_predictions(predictions_df, save_path='enhanced_prediction_analysis')

# Compute MSE and MAE
mse = mean_squared_error(predictions_df['Actual_Price'], predictions_df['Predicted_Price'])
mae = mean_absolute_error(predictions_df['Actual_Price'], predictions_df['Predicted_Price'])

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")