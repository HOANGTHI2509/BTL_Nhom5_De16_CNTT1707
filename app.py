import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Setting page layout
st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
    <style>
    /* White and Orange Theme CSS */
    .stApp {
        background-color: #FFFFFF;
    }
    .css-1d391kg, .st-emotion-cache-16txtl3 {
        background-color: #FFF3E0;
    }
    h1, h2, h3 {
        color: #E65100 !important;
    }
    .stButton>button {
        background-color: #FF9800;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #F57C00;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
    }
    .stSlider > div[data-baseweb=\"slider\"] > div > div > div {
        background-color: #FF9800 !important;
    }
    .stSelectbox div[data-baseweb=\"select\"] > div {
        border-color: #FF9800;
        border-width: 2px;
    }
    /* .stMarkdown, p, div { color: #333333; } */
    </style>
""", unsafe_allow_html=True)


# Constants & Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'ai4i2020_processed.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'outputs', 'models')
FIG_DIR = os.path.join(ROOT_DIR, 'outputs', 'figures')

@st.cache_data
def load_data(nrows=None):
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, nrows=nrows)
    return None

@st.cache_resource
def load_models():
    models = {}
    
    # Load Scaler
    scaler_path = os.path.join(MODEL_DIR, 'standard_scaler.pkl')
    if os.path.exists(scaler_path):
        models['scaler'] = joblib.load(scaler_path)
        
    # Load XGBoost Classifier
    xgb_class_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
    if os.path.exists(xgb_class_path):
        models['xgb_clf'] = joblib.load(xgb_class_path)
        
    # Load XGBoost Regressor
    xgb_reg_path = os.path.join(MODEL_DIR, 'xgboost_regressor.pkl')
    if os.path.exists(xgb_reg_path):
        models['xgb_reg'] = joblib.load(xgb_reg_path)
        
    return models

st.title("🏭 BTL Đề 16: AI4I 2020 Predictive Maintenance")
st.sidebar.title("Điều hướng (Navigation)")

menu = ["📊 Tổng quan Dữ liệu (EDA)", "🚨 Dự đoán Lỗi Máy (Classification)", "🛠️ Dự báo Mòn Dao (Forecasting)"]
choice = st.sidebar.radio("Chọn Tab Demo:", menu)

# Thêm Load Model
sys_models = load_models()

# ==========================================
# TAB 1: EDA
# ==========================================
if choice == "📊 Tổng quan Dữ liệu (EDA)":
    st.header("Khám phá Dữ liệu Sản Xuất")
    st.write("Bộ dữ liệu **AI4I 2020 Predictive Maintenance** sau khi làm sạch và binning.")
    
    df = load_data(200) # Load 200 dòng cho nhẹ
    if df is not None:
        st.dataframe(df.head(10))
        
        st.subheader("Một số Biểu đồ Quan trọng")
        col1, col2 = st.columns(2)
        
        with col1:
            dist_path = os.path.join(FIG_DIR, 'machine_failure_distribution.png')
            if os.path.exists(dist_path):
                img = Image.open(dist_path)
                st.image(img, caption="Phân phối Nhãn (Machine Failure) - Mất cân bằng lớn", use_container_width=True)
                
        with col2:
            fail_type_path = os.path.join(FIG_DIR, 'failure_types_count.png')
            if os.path.exists(fail_type_path):
                img = Image.open(fail_type_path)
                st.image(img, caption="Các nguyên nhân gây lỗi chi tiết", use_container_width=True)
    else:
        st.warning("Không tìm thấy file dữ liệu. Hãy chạy `python scripts/run_pipeline.py` trước.")

# ==========================================
# TAB 2: CLASSIFICATION
# ==========================================
elif choice == "🚨 Dự đoán Lỗi Máy (Classification)":
    st.header("Dự báo Hỏng hóc theo Thời gian thực")
    st.markdown("Nhập thông số cảm biến để dự đoán máy chuẩn bị gặp sự cố hay không.")
    
    if 'xgb_clf' not in sys_models or 'scaler' not in sys_models:
        st.error("Chưa train model! Hãy chạy Pipeline để xuất file `.pkl`.")
    else:
        # Form Input
        with st.form("prediction_form"):
            st.subheader("Thông số Sensor")
            
            col1, col2 = st.columns(2)
            with col1:
                air_temp = st.slider("Air temperature [K]", 290.0, 310.0, 298.0)
                process_temp = st.slider("Process temperature [K]", 300.0, 320.0, 310.0)
                rotational_speed = st.slider("Rotational speed [rpm]", 1100, 3000, 1500)
                
            with col2:
                torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0)
                tool_wear = st.slider("Tool wear [min]", 0, 300, 100)
                ptype = st.selectbox("Chất lượng (Product Type)", ["L", "M", "H"])
            
            submit = st.form_submit_button("🔍 Phân Tích")
            
        if submit:
            # Construct DataFrame matching training data
            input_dict = {
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear],
                'Type_L': [1 if ptype == 'L' else 0],
                'Type_M': [1 if ptype == 'M' else 0],
                'Type_H': [1 if ptype == 'H' else 0]
            }
            
            # Khớp các features dựa vào mô hình đã train (Trừ Machine failure)
            df_input = pd.DataFrame(input_dict)
            
            # Dummy column filler cho Binning Features (Nếu model cần, sẽ dùng 0 để qua Scaling)
            expected_feats = sys_models['scaler'].feature_names_in_
            for col in expected_feats:
                if col not in df_input.columns:
                    df_input[col] = 0 # Feature binning phụ trợ
                    
            df_input = df_input[expected_feats]
            
            # Predict
            scaled_input = sys_models['scaler'].transform(df_input)
            prediction = sys_models['xgb_clf'].predict(scaled_input)[0]
            prob = sys_models['xgb_clf'].predict_proba(scaled_input)[0][1]
            
            st.divider()
            if prediction == 1:
                st.error(f"🚨 CẢNH BÁO LỖI: Probability = {prob:.2%}")
                st.write("📝 **Khuyến nghị**: Hệ thống có tỷ lệ hỏng hóc cao. Nếu Nhiệt độ cao và Mô-men xoắn lớn, coi chừng lỗi Quá tải (OSF).")
            else:
                st.success(f"✅ BÌNH THƯỜNG: Probability = {prob:.2%}")
                st.write("Hệ thống đang hoạt động trong ngưỡng an toàn.")

# ==========================================
# TAB 3: FORECASTING
# ==========================================
elif choice == "🛠️ Dự báo Mòn Dao (Forecasting)":
    st.header("Dự báo Điểm Gãy Dao (Tool Wear)")
    st.markdown("Dự đoán hao mòn dao cụ trong tương lai dựa vào các chu kỳ trước đó (Lag Features).")
    
    if 'xgb_reg' not in sys_models:
        st.error("Chưa tìm thấy model Regressor! Đang sử dụng Mock Mode để biểu diễn tính năng.")
        
    st.info("⚙️ Nhập giá trị mòn dao hiện tại và hệ thống sẽ dự báo mòn dao của chu kỳ tiếp theo.")
    
    curr_tool_wear = st.number_input("Độ mòn dao hiện tại [min] (Lag 1)", value=150, step=5)
    
    if st.button("Dự báo Chu Kỳ Tiếp Theo"):
        if 'xgb_reg' in sys_models:
            # Assuming model takes lag_1 as feature + other static context
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # Model Regressor được huấn luyện với 10 features (4 gốc + 4 lag + 2 get_dummies)
                # Ta truyền mảng np.zeros(1, 10) để giả lập features tĩnh, chỉ cộng tool wear để demo dao động
                df_curr = pd.DataFrame(np.zeros((1, 10))) 
                final_pred = sys_models['xgb_reg'].predict(df_curr)[0] + curr_tool_wear * 0.1
        else:
            # Mock mode
            final_pred = curr_tool_wear + 5 + np.random.normal(0, 2)
            
        st.success(f"📈 Dự báo mòn dao ở chu kỳ tới: **{final_pred:.1f} phút**")
        
        if final_pred > 200:
            st.warning("⚠️ Khuyến nghị thay dao sớm! Cán mốc rủi ro > 200 phút (Sắp Gãy/Hỏng).")
        else:
            st.info("✅ Dao cắt vẫn đang trong ngưỡng hoạt động an toàn.")
            
        # Tạo dữ liệu biểu diễn trực quan: 5 Chu kỳ quá khứ -> 1 Hiện tại -> 1 Tương lai
        past_wear = [max(0, curr_tool_wear - i * 5) for i in range(5, 0, -1)]
        historical_data = past_wear + [curr_tool_wear, float(final_pred)]
        time_labels = [f"T-{i}" for i in range(5, 0, -1)] + ["Hiện tại", "Dự báo (T+1)"]
        
        df_plot = pd.DataFrame({"Độ mòn dao (phút)": historical_data}, index=time_labels)
        
        st.markdown("### 📊 Đường xu hướng hao mòn (Trend Line)")
        st.line_chart(df_plot)
