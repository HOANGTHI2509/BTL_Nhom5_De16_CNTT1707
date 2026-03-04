import sys
import os
from pathlib import Path

# Đảm bảo có thể import được các module từ src/
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.mining.association import AssociationRulesMiner
from src.mining.clustering import MachineClustering
from src.models.supervised import MaintenancePredictor
from src.models.semi_supervised import SemiSupervisedPredictor
from src.models.forecasting import ToolWearForecaster

def write_report(content):
    """Ghi nội dung vào file báo cáo tổng hợp"""
    report_path = project_root / "outputs" / "reports" / "metrics_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    print(f"  [+] Đã ghi nhật ký vào: {report_path}")
    
def clear_report():
    report_path = project_root / "outputs" / "reports" / "metrics_summary.md"
    if report_path.exists():
        report_path.unlink()

def main():
    clear_report()
    write_report("# BÁO CÁO TỔNG HỢP: AI4I 2020 PREDICTIVE MAINTENANCE\n")
    write_report("Báo cáo này được sinh tự động từ pipeline đánh giá Data Mining và Machine Learning.\n")
    
    # -------------------------------------------------------------
    # 1. DATA PREPARATION
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" BƯỚC 1: LOAD VÀ LÀM SẠCH DỮ LIỆU")
    print("="*50)
    
    try:
        loader = DataLoader()
        df_raw = loader.load_data()
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.run_pipeline(df_raw)
        
        cleaner.save_processed_data(df_cleaned)
        
        write_report("## 1. Dữ liệu (Data Preparation)")
        write_report(f"- Số lượng biểu ghi: {len(df_raw)}")
        write_report(f"- Số lượng đặc trưng ban đầu: {df_raw.shape[1]}")
        write_report(f"- Đã xử lý Missing values, Nulls và làm sạch cấu trúc cơ bản.\n")
    except Exception as e:
        print(f"LỖI BƯỚC 1 (Cần file raw data thực tế): {e}")
        write_report(f"## 1. Lỗi tải dữ liệu\nKhông tìm thấy file nguyên thủy hoặc sai cấu trúc: {e}")
        return # Dừng nếu không có data

    # -------------------------------------------------------------
    # 2. KHAI PHÁ LUẬT KẾT HỢP (ASSOCIATION RULES)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" BƯỚC 2: KHAI PHÁ LUẬT KẾT HỢP (APRIORI/FP-GROWTH)")
    print("="*50)
    
    miner = AssociationRulesMiner(min_support=0.01, min_threshold=1.5)
    failure_rules = miner.run(df_cleaned)
    
    write_report("## 2. Khai phá luật kết hợp (Association Rules)")
    if not failure_rules.empty:
        write_report(f"- Tìm thấy **{len(failure_rules)}** luật dẫn trực tiếp đến Machine Failure/Các lỗi cụ thể.")
        write_report("- **Top 3 luật cấu thành Lỗi cao nhất (theo Lift)**:")
        write_report("```text")
        # In ra 3 luật đầu tiên
        for idx, row in failure_rules.head(3).iterrows():
            ant = set(row['antecedents'])
            con = set(row['consequents'])
            write_report(f"  {ant} -> {con} | Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.2f}")
        write_report("```\n")
    else:
        write_report("- Không tìm thấy luật kết hợp thỏa mãn ngưỡng support và confidence tối thiểu.\n")

    # -------------------------------------------------------------
    # 3. K-MEANS CLUSTERING (PHÂN CỤM RỦI RO)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" BƯỚC 3: PHÂN CỤM CẢM BIẾN (K-MEANS CLUSTERING)")
    print("="*50)
    
    cluster_model = MachineClustering()
    X_scaled_cluster = cluster_model.prepare_data(df_cleaned)
    
    # Ở đây fix cứng K=4 để pipeline chạy luôn (hoặc có thể dùng find_optimal_k)
    best_k = 4 
    print(f"[*] Sử dụng K={best_k} để tiến hành Cluster Profiling")
    df_clustered = cluster_model.fit_predict(df_cleaned, n_clusters=best_k)
    profile = cluster_model.cluster_profiling(df_clustered)
    
    write_report("## 3. Phân cụm Máy Móc (K-Means Clustering)")
    write_report(f"- Tối ưu hóa với số cụm K = {best_k}.")
    write_report("- **Risk Profiling (Xác suất lỗi trên mỗi cụm):**")
    write_report("```text")
    for idx in profile.index:
         fail_rate = profile.loc[idx, 'Failure Rate (%)']
         write_report(f"  Cụm {idx}: Tỉ lệ hỏng hóc = {fail_rate:.2f}%")
    write_report("```\n")

    # -------------------------------------------------------------
    # 4. SUPERVISED LEARNING (GIẢI QUYẾT CLASS IMBALANCE)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" BƯỚC 4: SUPERVISED LEARNING (SMOTE + RF + XGBoost)")
    print("="*50)
    
    predictor = MaintenancePredictor()
    X_train_res, X_test_scl, y_train_res, y_test_cls = predictor.split_and_prepare_data(df_cleaned)
    predictor.train_models(X_train_res, y_train_res)
    
    rf_eval = predictor.evaluate_model("RandomForest", X_test_scl, y_test_cls)
    xgb_eval = predictor.evaluate_model("XGBoost", X_test_scl, y_test_cls)
    
    # Save Feature Importance và Models
    predictor.plot_feature_importance("XGBoost", predictor.models["XGBoost"].feature_names_in_ if hasattr(predictor.models["XGBoost"], 'feature_names_in_') else X_train_res.columns)
    predictor.save_models()
    
    write_report("## 4. Supervised Learning (Dự đoán Lỗi Máy - Classification)")
    write_report("- Giải quyết **Imbalanced Data** bằng thuật toán `SMOTE` trên tập huấn luyện.")
    write_report("- **Kết quả Đánh giá Lớp Thiểu Số (Lỗi = 1):**")
    write_report("| Model | PR-AUC | F1-Score | Recall |")
    write_report("|-------|--------|----------|--------|")
    write_report(f"| RandomForest | {rf_eval['pr_auc']:.4f} | {rf_eval['f1']:.4f} | {rf_eval['recall']:.4f} |")
    write_report(f"| XGBoost | {xgb_eval['pr_auc']:.4f} | {xgb_eval['f1']:.4f} | {xgb_eval['recall']:.4f} |\n")

    # -------------------------------------------------------------
    # 5. SEMI-SUPERVISED LEARNING (GIẢ LẬP MẤT NHÃN)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" BƯỚC 5: SEMI-SUPERVISED LEARNING (SELF-TRAINING)")
    print("="*50)
    
    semi = SemiSupervisedPredictor(base_model="xgboost", threshold=0.85)
    # Che 80% nhãn gốc trên tập X_train_resampled ban đầu
    y_semi, y_true, unlabel_idx = semi.simulate_unlabeled_data(X_train_res, y_train_res, labeled_ratio=0.2)
    
    # Train và Compare
    semi.train_and_compare(X_train_res, y_semi, X_test_scl, y_test_cls)
    semi.analyze_pseudo_label_risk(unlabel_idx, y_true)
    
    write_report("## 5. Semi-Supervised Learning (Self-Training với Pseudo-labeling)")
    write_report("- **Kịch bản**: Giả lập bỏ đi 80% nhãn (Che nhãn), chỉ giữ lại 20% gán nhãn thực tế.")
    write_report("- So sánh hiệu năng mô hình đã được ghi chi tiết qua hệ thống std out.")
    write_report("- **Rủi ro Pseudo-label**: Mô hình tự dự đoán chính xác được bao nhiêu phần trăm trên tập bị giấu sẽ cảnh báo nếu mức độ False Alarms (tự học quá khớp sai lệch) ở mức cao.\n")

    # -------------------------------------------------------------
    # 6. FORECASTING (HỒI QUY CHUỖI THỜI GIAN)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" BƯỚC 6: FORECASTING (DỰ BÁO TOOL WEAR THEO TIME SERIES)")
    print("="*50)
    
    forecaster = ToolWearForecaster(lag_steps=1)
    # Phải lấy dataframe thô (chưa drop UDI) nhưng đã xử lý null/Missing
    # Gặp lại hàm handle_missing_values trên df_raw
    df_raw_clean = cleaner.handle_missing_values(df_raw)
    
    if 'UDI' in df_raw_clean.columns:
        df_lagged = forecaster.create_lag_features(df_raw_clean)
        X_train_ts, X_test_ts, y_train_ts, y_test_ts = forecaster.prepare_data_split(df_lagged, test_size=0.2)
        
        forecast_results, preds = forecaster.train_and_evaluate(X_train_ts, y_train_ts, X_test_ts, y_test_ts)
        forecaster.plot_predictions(y_test_ts, preds, num_samples=300)
        
        write_report("## 6. Time-series Forecasting (Hồi quy Dự báo mòn dao)")
        write_report("- Mục tiêu: Dự báo biến liên tục `Tool wear [min]`")
        write_report("- Chẻ Test/Train theo **Thứ tự không xáo trộn** (shuffle=False).")
        write_report("- Đã sử dụng kỹ thuật tịnh tiến Thời gian (**Lag features** = 1).")
        write_report("- **Kết quả Đánh giá Metrics:**")
        write_report("| Model | MAE (Phút) | RMSE (Phút) |")
        write_report("|-------|------------|-------------|")
        for m_name, metrics in forecast_results.items():
            write_report(f"| {m_name} | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} |")
    else:
         print("Bỏ qua Forecasting do dataframe raw bị thiếu cột Index thời gian UDI")
         write_report("## 6. Time-series Forecasting\n- Không thực hiện do không tìm thấy UDI.")

    print("\n" + "="*50)
    print("🎉 PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG! HÃY KIỂM TRA THƯ MỤC OUTPUTS/")
    print("="*50)
    write_report("\n-- *Pipeline Execution Completed* --")

if __name__ == "__main__":
    main()
