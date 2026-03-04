import pandas as pd
from pathlib import Path
import yaml

class DataCleaner:
    def __init__(self, config_path="configs/params.yaml"):
        """Khởi tạo DataCleaner và lưu trữ đường dẫn cấu hình."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self):
        """Đọc file cấu hình params.yaml"""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return None

    def handle_missing_values(self, df):
        """Xử lý giá trị bị thiếu nếu có."""
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"[*] Phát hiện {missing_count} giá trị bị thiếu (missing values). Đang tiến hành xử lý...")
            # Sử dụng forward fill để điền dữ liệu khuyết thiếu, sau đó drop những cột không fill được
            df = df.fillna(method='ffill').dropna()
        else:
            print("[*] Không phát hiện giá trị bị thiếu (missing values).")
        return df

    def drop_unnecessary_columns(self, df):
        """Loại bỏ các cột không giúp ích cho quá trình huấn luyện/khai phá dữ liệu."""
        # UDI và Product ID thường là mã định danh, không có giá trị phân tích hoặc huấn luyện
        cols_to_drop = ['UDI', 'Product ID']
        existing_cols = [col for col in cols_to_drop if col in df.columns]
        
        df = df.drop(columns=existing_cols)
        print(f"[*] Đã loại bỏ các cột không cần thiết: {existing_cols}")
        return df

    def bin_continuous_variables(self, df):
        """Rời rạc hóa (binning) các biến liên tục thành nhãn 'Low', 'Medium', 'High'"""
        continuous_cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        
        df_binned = df.copy()
        for col in continuous_cols:
            if col in df_binned.columns:
                try:
                    # Rời rạc hóa sử dụng pd.qcut (chia dựa vào tứ phân vị - đều nội dung)
                    df_binned[f'{col}_binned'] = pd.qcut(
                        df_binned[col], 
                        q=3, 
                        labels=['Low', 'Medium', 'High'], 
                        duplicates='drop'
                    )
                except ValueError:
                    # Fallback dùng pd.cut cho trường hợp dữ liệu có quá nhiều đỉnh phân phối giống nhau
                    df_binned[f'{col}_binned'] = pd.cut(
                        df_binned[col], 
                        bins=3, 
                        labels=['Low', 'Medium', 'High']
                    )
                print(f"  -> Đã binning cột '{col}' thành '{col}_binned' (Low, Medium, High)")
                
        return df_binned

    def save_processed_data(self, df):
        """Lưu lại dữ liệu đã qua tiền xử lý ra thư mục data/processed/"""
        if not self.config:
            print("Cảnh báo: Không thể nạp được cấu hình.")
            return None
            
        processed_data_path = self.config.get("data_paths", {}).get("processed_data")
        if not processed_data_path:
            raise KeyError("Không tìm thấy đường dẫn tại 'data_paths.processed_data' trong params.yaml")
            
        project_root = self.config_path.parent.parent
        full_path = project_root / processed_data_path
        
        # Đảm bảo rẳng folder 'data/processed' đã tồn tại 
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(full_path, index=False)
        print(f"✅ Đã lưu dữ liệu qua tiền xử lý tại: {full_path}")
        return full_path
        
    def run_pipeline(self, df):
        """Chạy tổng thể pipeline làm sạch dữ liệu"""
        print("Bắt đầu Data Cleaning Pipeline...")
        df = self.handle_missing_values(df)
        df = self.drop_unnecessary_columns(df)
        df = self.bin_continuous_variables(df)
        print("✅ Hoàn thành quy trình làm sạch dữ liệu.")
        return df

if __name__ == "__main__":
    # Test DataCleaner bằng dữ liệu dummy (bỏ comments dòng dưới sau khi đã nạp dữ liệu thật)
    pass
