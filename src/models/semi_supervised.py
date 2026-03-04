import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SemiSupervisedPredictor:
    def __init__(self, base_model="xgboost", threshold=0.85):
        """
        Khởi tạo Semi-Supervised Predictor dựa trên Self-Training
        :param base_model: "xgboost" hoặc "rf"
        :param threshold: Ngưỡng tin cậy (Probability treshold) để gán pseudo-label
        """
        self.threshold = threshold
        self.base_model_name = base_model
        
        # XGBoost cần thiết lập probability để dùng trong SelfTraining
        if base_model == "xgboost":
            self.base_estimator = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        else:
            self.base_estimator = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            )
            
        # Mô hình Self Training
        self.semi_model = SelfTrainingClassifier(
            estimator=self.base_estimator,
            threshold=self.threshold,
            criterion='threshold',
            max_iter=10 # Số lần lặp lại tối đa để gán nhãn
        )

    def simulate_unlabeled_data(self, X_train, y_train, labeled_ratio=0.2, random_state=42):
        """
        Giả lập tình huống thiếu nhãn: 
        Giữ lại `labeled_ratio` nhãn thật, phần còn lại bị gán thành -1 (Unlabeled).
        Trả về Mảng Y_train đã bị che, và Y_train gốc để phân tích sau này (cho 80% kia).
        """
        print(f"\n[*] Đang giả lập kịch bản Semi-supervised: Chỉ giữ {labeled_ratio*100}% nhãn gốc.")
        rng = np.random.RandomState(random_state)
        
        # Mảng nhãn thực sự để check
        y_true_hidden = np.array(y_train, copy=True)
        
        # Mảng nhãn sẽ bị che (-1) để đem vào SelfTraining
        y_train_semi = np.array(y_train, copy=True)
        
        n_samples = len(y_train)
        n_labeled = int(labeled_ratio * n_samples)
        
        # Chọn ngẫu nhiên các index được giữ lại nhãn
        labeled_indices = rng.choice(n_samples, n_labeled, replace=False)
        
        # Tạo mask nhãn: 0/1 đối với index đã chọn, -1 cho phần còn lại
        unlabeled_indices = np.setdiff1d(np.arange(n_samples), labeled_indices)
        y_train_semi[unlabeled_indices] = -1
        
        print(f"  -> Tổng mẫu Train: {n_samples}")
        print(f"  -> Nhãn được giữ (Labeled): {n_labeled}")
        print(f"  -> Bị che nhãn (Unlabeled = -1): {len(unlabeled_indices)}")
        
        return y_train_semi, y_true_hidden, unlabeled_indices

    def _evaluate_metrics(self, y_test, y_pred, y_prob):
        """Hàm helper tính PR-AUC và F1"""
        f1 = f1_score(y_test, y_pred)
        precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recalls, precisions)
        return f1, pr_auc

    def train_and_compare(self, X_train, y_train_semi, X_test, y_test):
        """
        So sánh mô hình (1) Supervised-only (Chỉ train bằng 20% nhãn có sẵn) 
        với (2) Semi-supervised (Train bằng 20% nhãn + 80% pseudo-labels)
        """
        # 1. Huấn luyện Supervised Only (Base model) chỉ trên tập 20%
        print("\n[*] Đang huấn luyện mô hình cơ sở (Supervised) CHỈ với dữ liệu có nhãn...")
        labeled_mask = (y_train_semi != -1)
        X_labeled = X_train[labeled_mask]
        y_labeled = y_train_semi[labeled_mask]
        
        self.base_estimator.fit(X_labeled, y_labeled)
        
        y_pred_base = self.base_estimator.predict(X_test)
        y_prob_base = self.base_estimator.predict_proba(X_test)[:, 1]
        f1_base, pr_auc_base = self._evaluate_metrics(y_test, y_pred_base, y_prob_base)
        print(f"  -> Base Model - F1-Score: {f1_base:.4f} | PR-AUC: {pr_auc_base:.4f}")
        
        # 2. Huấn luyện Semi-supervised Self-Training bằng toàn bộ (Labeled + Unlabeled)
        print("\n[*] Đang huấn luyện thuật toán Self-Training (Semi-supervised) bằng Dữ liệu Labeled + Unlabeled...")
        self.semi_model.fit(X_train, y_train_semi)
        
        y_pred_semi = self.semi_model.predict(X_test)
        y_prob_semi = self.semi_model.predict_proba(X_test)[:, 1]
        f1_semi, pr_auc_semi = self._evaluate_metrics(y_test, y_pred_semi, y_prob_semi)
        print(f"  -> Semi-Supervised Model - F1-Score: {f1_semi:.4f} | PR-AUC: {pr_auc_semi:.4f}")
        
        print("\n=== Tổng kết hiệu suất ===")
        print(f"Sự cải thiện do Pseudo-label:")
        print(f"  - Tăng Δ F1-Score: {f1_semi - f1_base:+.4f}")
        print(f"  - Tăng Δ PR-AUC: {pr_auc_semi - pr_auc_base:+.4f}")
        
        return self.semi_model

    def analyze_pseudo_label_risk(self, unlabeled_indices, y_true_hidden):
        """
        Phân tích chất lượng các Pseudo-labels do SelfTraining Classifier sinh ra trong quá trình lặp.
        Kiểm tra độ chính xác của nhãn giả (so với nhãn gốc thực tế y_true_hidden đã được giữ bí mật)
        """
        print("\n[*] Đang phân tích rủi ro của thuật toán Pseudo-Labeling...")
        
        # Lấy nhãn cuối cùng từ mô hình sau tất cả các vòng lặp 
        # (self.semi_model.transduction_ chứa nhãn cho mọi mẫu trong X_train, kể cả mẫu ban đầu có id = -1)
        final_labels = self.semi_model.transduction_
        
        # Chỉ tập trung đánh giá sai số trên tập Unlabeled
        pseudo_labels = final_labels[unlabeled_indices]
        true_labels = y_true_hidden[unlabeled_indices]
        
        # Phân tích mức độ False Alarm (Model đoán hỏng = 1, thực tế bình thường = 0)
        false_alarms = np.sum((pseudo_labels == 1) & (true_labels == 0))
        
        # Phân tích mức độ Miss (Model đoán bình thường = 0, thực tế máy hỏng = 1)
        missed_failures = np.sum((pseudo_labels == 0) & (true_labels == 1))
        
        correct_preds = np.sum(pseudo_labels == true_labels)
        total_pseudo = len(pseudo_labels)
        
        pseudo_accuracy = correct_preds / total_pseudo
        
        print(f"=== Số liệu rủi ro trên tập Unlabeled ({total_pseudo} điểm) ===")
        print(f"  - Số nhãn giả đoán ĐÚNG: {correct_preds} ({pseudo_accuracy*100:.2f}%)")
        print(f"  - False Alarms (Báo giả lỗi máy trên tập Unlabeled): {false_alarms}")
        print(f"  - Missed Failures (Bỏ lót máy Hỏng trên tập Unlabeled): {missed_failures}")
        
        if false_alarms > 0:
            print("CẢNH BÁO: False alarms sinh ra từ pseudo-labeling có thể tạo ra hiệu ứng 'Bias Amplification' khiến thế hệ RF/XGBoost tự học cái sai của chính nó!")

if __name__ == "__main__":
    pass
    # # Test pipeline Semi-supervised 
    # # Giả định đã sử dụng predictor_supervised.split_and_prepare_data để có X_train_scaled/y_train
    # semi = SemiSupervisedPredictor(base_model="xgboost", threshold=0.85)
    # y_semi, y_true, unlabel_idx = semi.simulate_unlabeled_data(X_train_resampled, y_train_resampled)
    # model_semi = semi.train_and_compare(X_train_resampled, y_semi, X_test_scaled, y_test)
    # semi.analyze_pseudo_label_risk(unlabel_idx, y_true)
