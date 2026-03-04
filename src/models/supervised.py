import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, \
    precision_recall_curve, auc, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

class MaintenancePredictor:
    def __init__(self, config_path="configs/params.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = self.config_path.parent.parent
        self.scaler = StandardScaler()
        
        # Các mô hình sẽ được lưu trên class sau khi train
        self.models = {}
        
    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        raise FileNotFoundError(f"Không tìm thấy cấu hình {self.config_path}")

    def split_and_prepare_data(self, df):
        """
        Tách tập Train/Test và áp dụng Scale/SMOTE cho vấn đề Mất cân bằng dữ liệu.
        Chỉ Scale trên X_train để tránh hiện tượng rò rỉ dữ liệu (Data Leakage).
        """
        print("[*] Đang chuẩn bị dữ liệu (Train/Test Split & SMOTE)...")
        # Target: Dùng 'Machine failure' làm nhãn dự đoán chung
        target_col = 'Machine failure'
        
        # Đặc trưng (X): Bỏ tất cả các cột lỗi cụ thể không dùng làm Features 
        # (TWF, HDF, PWF, OSF, RNF) vì chúng trực tiếp gây ra label 1
        cols_to_drop = [target_col, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        # Lọc bỏ biến Rời rạc hóa (_binned) đã dùng cho association
        # Chỉ lấy đặc trưng gốc & Category 'Type' (One-hot encode)
        drop_binned = [col for col in df.columns if col.endswith('_binned')]
        cols_to_drop.extend(drop_binned)
        
        # Đảm bảo columns tồn tại thì mới drop
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        
        X = df.drop(columns=cols_to_drop)
        
        # One-Hot Encoding cho cột 'Type' nếu có
        if 'Type' in X.columns:
            X = pd.get_dummies(X, columns=['Type'], drop_first=True)
            
        y = df[target_col]
        
        test_size = self.config.get("train_test_split_ratio", 0.2)
        random_seed = self.config.get("random_seed", 42)
        
        # Phân tầng (stratify=y) để giữ đúng tỉ lệ nhãn Imbalance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        
        # Standard Scaler (Chỉ Fit trên Train)
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
        
        # Áp dụng SMOTE để xử lý mất cân bằng lớp
        smote = SMOTE(random_state=random_seed)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        print(f"  -> Kích thước tập Train ban đầu: Cấp 0: {sum(y_train==0)}, Cấp 1: {sum(y_train==1)}")
        print(f"  -> Sau SMOTE (Train): Cấp 0: {sum(y_train_resampled==0)}, Cấp 1: {sum(y_train_resampled==1)}")
        print(f"  -> Kích thước tập Test: Cấp 0: {sum(y_test==0)}, Cấp 1: {sum(y_test==1)}")
        
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test

    def train_models(self, X_train, y_train):
        """Huấn luyện mô hình RandomForest và XGBoost"""
        rf_params = self.config.get("hyperparameters", {}).get("RandomForest", {})
        xgb_params = self.config.get("hyperparameters", {}).get("XGBoost", {})
        
        print("\n[*] Đang huấn luyện Random Forest...")
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        
        print("[*] Đang huấn luyện XGBoost...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
        xgb_model.fit(X_train, y_train)
        self.models["XGBoost"] = xgb_model
        
        return self.models

    def evaluate_model(self, model_name, X_test, y_test):
        """
        Đánh giá mô hình Mất Cân Bằng.
        KHÔNG DÙNG ACCURACY. Dùng: PR-AUC, F1-Score (Minority Class), Recall.
        In và lưu Confusion Matrix.
        """
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Chưa huấn luyện mô hình {model_name}")
            
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # Xác suất lớp 1 (Hỏng)
        
        # Các chỉ số cho lớp thiểu số (Minority Class 1 - Hỏng máy)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # PR-AUC
        precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recalls, precisions)
        
        print(f"\n========== Đánh giá {model_name} ==========")
        print(f"Recall (Độ nhạy - Bắt lỗi thành công): {recall:.4f}")
        print(f"Precision (Độ chính xác - Báo động đúng): {precision:.4f}")
        print(f"F1-Score (Cân bằng P/R): {f1:.4f}")
        print(f"PR-AUC (Area Under Precision-Recall Curve): {pr_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self._plot_confusion_matrix(y_test, y_pred, model_name)
        
        return {"recall": recall, "f1": f1, "pr_auc": pr_auc}

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Norm (0)', 'Fail (1)'],
                    yticklabels=['Norm (0)', 'Fail (1)'])
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        fig_path = self.project_root / "outputs" / "figures" / f"cm_{model_name}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300)
        plt.close()

    def plot_feature_importance(self, model_name, feature_names):
        """Vẽ độ quan trọng của các đặc trưng (Variable Importance)"""
        model = self.models.get(model_name)
        if not model:
            print(f"Không thể vẽ Feature Importance: Model {model_name} chưa tồn tại.")
            return
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        num_features = len(feature_names)
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(num_features), importances[indices], align="center")
        plt.xticks(range(num_features), np.array(feature_names)[indices], rotation=90)
        plt.xlim([-1, num_features])
        plt.tight_layout()
        
        fig_path = self.project_root / "outputs" / "figures" / f"feature_importance_{model_name}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Đã lưu biểu đồ Feature Importance của {model_name} vào outputs/figures/")

    def save_models(self):
        """Xuất mô hình ra file .pkl để triển khai sau này"""
        models_dir = self.project_root / "outputs" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            path = models_dir / f"{name.lower()}_model.pkl"
            joblib.dump(model, path)
            print(f"✅ Đã lưu mô hình: {path}")
            
        # Lưu StandardScaler sử dụng cho Inferencing tương lai
        scaler_path = models_dir / "standard_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

if __name__ == "__main__":
    pass
    # # Pseudo test logic
    # # Đọc dữ liệu từ df_cleaned
    # predictor = MaintenancePredictor()
    # X_train, X_test, y_train, y_test = predictor.split_and_prepare_data(df_cleaned)
    # predictor.train_models(X_train, y_train)
    # predictor.evaluate_model("RandomForest", X_test, y_test)
    # predictor.evaluate_model("XGBoost", X_test, y_test)
    # predictor.plot_feature_importance("RandomForest", X_train.columns)
    # predictor.save_models()
