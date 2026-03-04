import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

class AssociationRulesMiner:
    def __init__(self, min_support=0.01, min_threshold=1.0, metric='lift'):
        """
        Khởi tạo Association Rules Miner (sử dụng Apriori Algorithm)
        :param min_support: Ngưỡng Support tối thiểu cho tập phổ biến
        :param min_threshold: Ngưỡng metric tối thiểu (phụ thuộc vào metric)
        :param metric: Chỉ số dùng để đánh giá độ tin cậy của luật (lift, confidence, support...)
        """
        self.min_support = min_support
        self.min_threshold = min_threshold
        self.metric = metric

    def prepare_transaction_data(self, df):
        """
        Chuẩn bị dữ liệu đầu vào cho mlxtend: Chuyển đổi dữ liệu rời rạc (binned data) 
        sang one-hot encoding (boolean).
        """
        # Lọc ra các cột binned (Low, Medium, High) và cột target lỗi (Machine failure, TWF, HDF, PWF, OSF, RNF)
        binned_cols = [col for col in df.columns if col.endswith('_binned')]
        
        # Các cột target (chuyển giá trị 1 thành nhãn kiểu chuỗi để phân biệt)
        target_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        
        # Dataframe chỉ chứa các đặc trưng sử dụng cho phân tích luật kết hợp
        df_selected = df[binned_cols + target_cols].copy()
        
        # Đổi tên cột _binned cho đẹp
        df_selected = df_selected.rename(columns={col: col.replace('_binned', '') for col in binned_cols})
        
        # Chuyển các cột target từ số (0, 1) sang tên nhãn để dễ nhìn trong luật.
        # Giá trị 0 (bình thường) sẽ drop ra khỏi One-Hot vì ta chỉ quan tâm luật gây ra LỖI (1)
        for t_col in target_cols:
            df_selected[t_col] = df_selected[t_col].apply(lambda x: t_col if x == 1 else 'No_Issue')
            
        # One-Hot Encoding: Pandas get_dummies
        df_encoded = pd.get_dummies(df_selected)
        
        # Loại bỏ các cột 'No_Issue' vì ta không cần tìm luật dẫn đến việc "không có lỗi"
        cols_to_drop = [col for col in df_encoded.columns if 'No_Issue' in col]
        df_encoded = df_encoded.drop(columns=cols_to_drop)
        
        # Chuyển đổi toàn bộ sang boolean theo yêu cầu của phiên bản mlxtend mới nhất
        df_encoded = df_encoded.astype(bool)
        return df_encoded

    def mine_rules(self, df_encoded):
        """Khai thác các tập phổ biến và sinh luật kết hợp"""
        print(f"[*] Đang tìm các tập phổ biến (Frequent Itemsets) với min_support={self.min_support}")
        
        # Apriori Algorithm
        frequent_itemsets = apriori(df_encoded, min_support=self.min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            print("Cảnh báo: Không tìm thấy tập phổ biến nào thỏa mãn min_support!")
            return pd.DataFrame()
            
        print(f"[*] Đang tạo các Association Rules với {self.metric} >= {self.min_threshold}")
        rules = association_rules(frequent_itemsets, metric=self.metric, min_threshold=self.min_threshold)
        return rules

    def filter_rules_by_consequents(self, rules, target_consequents):
        """
        Lọc các luật MÀ HỆ QUẢ ĐẦU RA (Consequents) là các loại lỗi (Machine failure, TWF...).
        """
        if rules.empty:
            return rules
            
        # target_consequents là list các chuỗi, ví dụ: ['Machine failure_Machine failure']
        # Do One-Hot nên tên cột Pandas dummy lúc nãy sẽ ghép lại như trên.
        target_set = set(target_consequents)
        
        def has_target_consequent(consequents_frozenset):
            # Kiểm tra xem giao của tập hệ quả với tập kết quả ta mong muốn có rỗng không
            return not frozenset(target_set).isdisjoint(consequents_frozenset)
            
        # Lọc các dòng rule có consequent nằm trong target
        filtered_rules = rules[rules['consequents'].apply(has_target_consequent)]
        
        # Sắp xếp luật theo Lift giảm dần, sau đó đến Confidence giảm dần
        filtered_rules = filtered_rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
        return filtered_rules

    def run(self, df):
        """Thực thi Pipeline Association Rules"""
        df_encoded = self.prepare_transaction_data(df)
        rules = self.mine_rules(df_encoded)
        
        target_failures = [
             'Machine failure_Machine failure',
             'TWF_TWF', 'HDF_HDF', 'PWF_PWF', 'OSF_OSF', 'RNF_RNF'
        ]
        
        failure_rules = self.filter_rules_by_consequents(rules, target_failures)
        
        print(f"✅ Đã tìm thấy {len(failure_rules)} luật dẫn đến hỏng hóc/lỗi.")
        return failure_rules

if __name__ == "__main__":
    pass
    # # Test Association Miner (Giả định đã có df_cleaned từ DataCleaner)
    # miner = AssociationRulesMiner(min_support=0.005, min_threshold=2.0)
    # failure_rules = miner.run(df_cleaned)
    # print(failure_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
