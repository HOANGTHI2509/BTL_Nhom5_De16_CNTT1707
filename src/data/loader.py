import pandas as pd
import yaml
from pathlib import Path

class DataLoader:
    def __init__(self, config_path="configs/params.yaml"):
        """
        Khởi tạo DataLoader với đường dẫn đến file cấu hình.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Danh sách các cột dự kiến dựa trên schema yêu cầu
        self.expected_columns = [
            "UDI", "Product ID", "Type", "Air temperature", "Process temperature", 
            "Rotational speed", "Torque", "Tool wear", 
            "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"
        ]

    def _load_config(self):
        """Đọc file params.yaml"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {self.config_path}")
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
            
    def validate_schema(self, df):
        """Kiểm tra xem dataframe có chứa đầy đủ các cột yêu cầu không"""
        # Dataset AI4I 2020 thực tế có kèm theo đơn vị đo lường trong tên cột, 
        # ví dụ: 'Air temperature [K]', 'Torque [Nm]'. 
        # Chúng ta có thể kiểm tra xem tên cột mong muốn có nằm trong tên cột thực tế không.
        
        missing_cols = []
        for expected_col in self.expected_columns:
            # Kiểm tra xem có cột nào chứa chuỗi expected_col không
            if not any(expected_col in col for col in df.columns):
                missing_cols.append(expected_col)
                
        if missing_cols:
            raise ValueError(f"Dữ liệu thiếu các cột (schema error): {missing_cols}")
            
        print("✅ Kiểm tra Schema thành công! Dữ liệu hợp lệ.")
        return True

    def load_data(self):
        """Đọc dữ liệu từ đường dẫn được cấu hình trong params.yaml"""
        # Lấy file đường dẫn gốc tương đối với vị trí hiện tại (root của dự án)
        project_root = self.config_path.parent.parent
        data_path = self.config.get("data_paths", {}).get("raw_data")
        
        if not data_path:
            raise KeyError("Không tìm thấy cấu hình 'data_paths.raw_data' trong params.yaml")
            
        full_path = project_root / data_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {full_path}. Hãy đảm bảo bạn đã đưa file dataset vào đây.")
            
        print(f"Loading data từ: {full_path}...")
        df = pd.read_csv(full_path)
        
        # Chuẩn hóa lại tên cột cho giống với yêu cầu (loại bỏ đơn vị đo lường [K], [rpm], [Nm], [min])
        renamed_columns = {
            col: col.replace(' [K]', '').replace(' [rpm]', '').replace(' [Nm]', '').replace(' [min]', '') 
            for col in df.columns
        }
        df = df.rename(columns=renamed_columns)
            
        self.validate_schema(df)
        return df

if __name__ == "__main__":
    # Test DataLoader
    try:
        loader = DataLoader()
        # df = loader.load_data() # Uncomment khi đã có file csv
    except Exception as e:
        print(e)
