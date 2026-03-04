import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

class ToolWearForecaster:
    def __init__(self, lag_steps=1):
        """
        Dự báo độ mòn dao cụ (Tool wear - Hồi quy) dựa trên UDI (thời gian xấp xỉ)
        :param lag_steps: Số bước lùi (lag) của cảm biến để dự báo tương lai
        """
        self.lag_steps = lag_steps
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.models = {
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
        }
        
    def create_lag_features(self, df):
        """
        Tạo các biến trễ (Lag features) cho chuỗi thời gian xấp xỉ UDI.
        Sử dụng trạng thái cảm biến ở thời điểm t-1 để dự đoán Tool wear ở thời điểm t.
        """
        print(f"[*] Đang tạo biến trễ (Lag = {self.lag_steps}) cho dữ liệu cảm biến...")
        df_forecasting = df.copy()
        
        # Sắp xếp theo UDI để đảm bảo đúng thứ tự thời gian
        if 'UDI' in df_forecasting.columns:
            df_forecasting = df_forecasting.sort_values(by='UDI').reset_index(drop=True)
            
        sensor_cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque']
        
        # Chỉ tạo lag cho các cột cảm biến có tồn tại
        sensors_exist = [col for col in sensor_cols if col in df_forecasting.columns]
        
        for col in sensors_exist:
            for lag in range(1, self.lag_steps + 1):
                df_forecasting[f'{col}_lag{lag}'] = df_forecasting[col].shift(lag)
                
        # Drop các dòng bị NaN do quá trình shift (tạo lag)
        df_forecasting = df_forecasting.dropna(subset=[f'{col}_lag1' for col in sensors_exist])
        
        return df_forecasting

    def prepare_data_split(self, df, test_size=0.2):
        """
        Tách Train/Test nhưng KHÔNG xáo trộn (shuffle=False) 
        để bảo toàn cấu trúc chuỗi thời gian của UDI.
        """
        target_col = 'Tool wear'
        
        # Đảm bảo bỏ các label phân loại và UDI/Product ID (không mang ý nghĩa tương lai)
        drop_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', target_col]
        # Bỏ luôn binned cols nếu còn
        drop_cols.extend([col for col in df.columns if col.endswith('_binned')])
        
        features_to_drop = [c for c in drop_cols if c in df.columns]
        
        X = df.drop(columns=features_to_drop)
        
        # Xử lý biến phân loại (Type)
        if 'Type' in X.columns:
            X = pd.get_dummies(X, columns=['Type'], drop_first=True)
            
        y = df[target_col].values.reshape(-1, 1)
        
        # Train/Test Time-Series Split (Chẻ dọc mảng, không bốc ngẫu nhiên)
        split_idx = int(len(X) * (1 - test_size))
        
        print(f"[*] Cắt Time-Series Train/Test (Tỷ lệ {1-test_size}:{test_size}) tại Index: {split_idx}")
        
        X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale Data (Fit trên Train, Transform trên Test)
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train).flatten()
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test.flatten()

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Huấn luyện và đánh giá trên tập Test bằng MAE và RMSE (đã inverse scale cho target)"""
        results = {}
        predictions_dict = {}
        
        for name, model in self.models.items():
            print(f"\n[*] Đang huấn luyện mô hình dự báo {name}...")
            model.fit(X_train, y_train)
            
            # Dự đoán (giá trị scale)
            y_pred_scaled = model.predict(X_test)
            
            # Khôi phục giá trị thực (Inverse Transform) cho dự đoán để tính toán Loss trên giá trị tool wear thực tế (phút)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            predictions_dict[name] = y_pred
            
            # Đánh giá Regression
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {'MAE': mae, 'RMSE': rmse}
            
            print(f"  -> {name} | MAE: {mae:.2f} (phút) | RMSE: {rmse:.2f} (phút)")
            
        return results, predictions_dict

    def plot_predictions(self, y_test, predictions_dict, num_samples=200):
        """
        Vẽ biểu đồ Line plot so sánh Thực tế vs Dự đoán của Tool wear.
        Để dễ nhìn, chỉ vẽ `num_samples` điểm dữ liệu cuối cùng của tập Test.
        """
        plt.figure(figsize=(15, 6))
        
        # Cắt lấy khúc đuôi để vẽ cho dễ nhìn
        actual = y_test[-num_samples:]
        plt.plot(actual, label='Thực tế (Actual Tool Wear)', color='black', linewidth=2)
        
        colors = ['red', 'blue']
        for i, (name, y_pred) in enumerate(predictions_dict.items()):
            pred_subset = y_pred[-num_samples:]
            plt.plot(pred_subset, label=f'Dự đoán ({name})', color=colors[i % len(colors)], linestyle='--', alpha=0.8)
            
        plt.title(f'Dự báo Độ mòn dao cụ (Tool Wear) - {num_samples} điểm cuối tập Test', fontsize=14)
        plt.xlabel('Thời gian (Index xấp xỉ)', fontsize=12)
        plt.ylabel('Tool Wear [phút]', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Lưu hình ảnh
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        fig_path = os.path.join(project_root, 'outputs', 'figures', 'tool_wear_forecast.png')
        
        if os.path.exists(os.path.dirname(fig_path)):
            plt.savefig(fig_path, dpi=300)
            print(f"\n✅ Đã lưu biểu đồ dự báo (Line plot) tại: {fig_path}")
            
        plt.show(block=False)
        plt.close()

if __name__ == "__main__":
    pass
    # # Test Workflow:
    # # Chú ý: Cần truyền df gốc (chưa qua DataCleaner drop column) hoặc đảm bảo df_cleaned có cột UDI.
    # # Nếu Cleaner_drop_UDI => Phải pass df_raw sau khi fill_na vào đây.
    # forecaster = ToolWearForecaster(lag_steps=1)
    # df_lagged = forecaster.create_lag_features(df_raw) # df_raw có 'UDI'
    # X_train, X_test, y_train, y_test = forecaster.prepare_data_split(df_lagged, test_size=0.2)
    # results, preds = forecaster.train_and_evaluate(X_train, y_train, X_test, y_test)
    # forecaster.plot_predictions(y_test, preds, num_samples=300)
