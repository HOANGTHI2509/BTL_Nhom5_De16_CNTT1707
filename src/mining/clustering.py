import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

class MachineClustering:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        
        # Các cảm biến liên tục dùng cho clustering
        self.features = [
            'Air temperature', 
            'Process temperature', 
            'Rotational speed', 
            'Torque', 
            'Tool wear'
        ]

    def prepare_data(self, df):
        """Chuẩn hóa dữ liệu đầu vào cho thuật toán K-Means"""
        X = df[self.features].copy()
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=self.features)

    def find_optimal_k(self, X_scaled, max_k=10, plot_elbow=True):
        """Tìm K tối ưu bằng phương pháp Elbow (Inertia) và Silhouette Score"""
        print(f"[*] Đang tìm số cụm K tốt nhất từ 2 đến {max_k}...")
        
        inertias = []
        silhouette_scores = []
        k_values = range(2, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)
            
            inertias.append(kmeans.inertia_)
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"  -> K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
            
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"✅ Dựa theo Silhouette Score lớn nhất, K tối ưu có thể là: {optimal_k}")
        
        if plot_elbow:
            self._plot_elbow_and_silhouette(k_values, inertias, silhouette_scores)
            
        return optimal_k

    def _plot_elbow_and_silhouette(self, k_values, inertias, silhouette_scores):
        """Lưu biểu đồ phân tích Elbow và Silhouette ra file"""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Số lượng cụm (K)')
        ax1.set_ylabel('Inertia (Elbow Method)', color=color)
        ax1.plot(k_values, inertias, marker='o', color=color, linewidth=2, label='Inertia')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Silhouette Score', color=color)  
        ax2.plot(k_values, silhouette_scores, marker='s', color=color, linewidth=2, label='Silhouette')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.title('Đánh giá số cụm tối ưu K-Means bằng Elbow & Silhouette')
        
        # Save plot
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        fig_path = os.path.join(project_root, 'outputs', 'figures', 'kmeans_optimal_k.png')
        if os.path.exists(os.path.dirname(fig_path)):
            plt.savefig(fig_path, dpi=300)
        plt.show(block=False)
        plt.close()

    def fit_predict(self, df, n_clusters):
        """Huấn luyện K-Means và gán nhãn cụm cho Dataset gốc"""
        X_scaled = self.prepare_data(df)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        return df_clustered

    def cluster_profiling(self, df_clustered):
        """
        Phân tích hồ sơ từng cụm: Trả về trung bình các giá trị cảm biến 
        và xác suất Hỏng hóc (Machine failure) của cụm đó.
        """
        print("\n=== CLUSTER PROFILING ===")
        # Gom nhóm theo Cluster và tính các giá trị đặc trưng (Giá trị gốc, chưa Scale)
        cols_to_profile = self.features + ['Machine failure']
        profile = df_clustered.groupby('Cluster')[cols_to_profile].mean()
        
        # Đổi Format phần chăn cho dễ đọc
        profile['Machine failure'] = profile['Machine failure'] * 100 # Thành %
        profile = profile.rename(columns={'Machine failure': 'Failure Rate (%)'})
        
        # Phân loại độ rủi ro của từng Cluster
        for idx in profile.index:
            failure_rate = profile.loc[idx, 'Failure Rate (%)']
            if failure_rate > 10:
                print(f"Cụm {idx}: ❌ Nguy cơ LỖI RẤT CAO (Tỷ lệ hỏng: {failure_rate:.1f}%)")
            elif failure_rate > 3:
                print(f"Cụm {idx}: ⚠️ Nguy cơ CẢNH BÁO (Tỷ lệ hỏng: {failure_rate:.1f}%)")
            else:
                print(f"Cụm {idx}: ✅ Máy hoạt động BÌNH THƯỜNG (Tỷ lệ hỏng: {failure_rate:.1f}%)")
                
        return profile

if __name__ == "__main__":
    pass
    # # Test KMeans Miner 
    # model = MachineClustering()
    # X_scaled = model.prepare_data(df_cleaned)
    # best_k = model.find_optimal_k(X_scaled, max_k=8)
    #
    # df_clustered = model.fit_predict(df_cleaned, n_clusters=best_k)
    # profile = model.cluster_profiling(df_clustered)
    # print(profile)
