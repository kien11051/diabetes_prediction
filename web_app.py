import streamlit as st
import pandas as pd
import joblib

from processing import encoding_data

# Cấu hình trang Web
st.set_page_config(page_title="Dự đoán Tiểu Đường", page_icon="🩺", layout="centered")

# --- 1. HÀM TẢI TÀI NGUYÊN (CÓ CACHE) ---
# @st.cache_resource giúp Web không phải load lại file .joblib 
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('diabetes_model.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, model_columns
    except Exception as e:
        st.error(f"Lỗi tải model: {e}. Vui lòng kiểm tra lại file .joblib")
        return None, None

model, model_columns = load_assets()

# --- 2. GIAO DIỆN NGƯỜI DÙNG ---
st.title("🩺 Hệ Thống Dự Đoán Nguy Cơ Tiểu Đường")
st.write("Vui lòng nhập các chỉ số sức khỏe của bạn để hệ thống đánh giá.")

# Tạo form nhập liệu bằng cách chia 2 cột 
col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin cá nhân")
    gender = st.selectbox("Giới tính", ["Male", "Female", "Other"])
    age = st.number_input("Tuổi", min_value=0, max_value=120, value=50, step=1)
    bmi = st.number_input("Chỉ số BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    smoking_history = st.selectbox(
        "Lịch sử hút thuốc", 
        ["never", "current", "former", "ever", "not current", "No Info"]
    )

with col2:
    st.subheader("Chỉ số y tế")
    hypertension_input = st.radio("Tiền sử Cao huyết áp", ["Không", "Có"])
    heart_disease_input = st.radio("Tiền sử Bệnh tim", ["Không", "Có"])
    hba1c = st.number_input("Chỉ số HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    glucose = st.number_input("Đường huyết (mg/dL)", min_value=50, max_value=400, value=100, step=1)

# Chuyển đổi input "Có/Không" thành 1/0
hypertension = 1 if hypertension_input == "Có" else 0
heart_disease = 1 if heart_disease_input == "Có" else 0

# --- 3. XỬ LÝ VÀ DỰ ĐOÁN ---
# Tạo nút bấm (Button)
if st.button("🔍 Phân Tích & Dự Đoán", use_container_width=True):
    if model is not None:
        # Tạo DataFrame từ input
        input_data = pd.DataFrame([{
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'hbA1c_level': hba1c,
            'blood_glucose_level': glucose
        }])

        # Tiền xử lý (One-hot & Reindex) - Giữ nguyên logic cũ
        input_encoded = encoding_data(input_data, categorical_cols=['gender', 'smoking_history'])
        final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Dự đoán
        probability = model.predict_proba(final_input)
        risk_score = probability[0][1] * 100

        # --- 4. HIỂN THỊ KẾT QUẢ ---
        st.markdown("---")
        st.subheader("📋 Kết Quả Đánh Giá")
        
        # Hiển thị điểm rủi ro to rõ ràng
        st.metric(label="Điểm rủi ro (AI đánh giá)", value=f"{risk_score:.1f}%")

        # Logic Y khoa 
        if risk_score >= 50:
            st.error("🔴 **CẢNH BÁO: NGUY CƠ CAO**\nMô hình dự đoán bạn có khả năng cao mắc bệnh tiểu đường. Hãy đến cơ sở y tế để xét nghiệm ngay.")
        elif (5.7 <= hba1c <= 6.5) or (100 <= glucose <= 125):
            st.warning(f"🟠 **CẢNH BÁO: TIỀN TIỂU ĐƯỜNG (Pre-diabetes)**\nMặc dù AI dự đoán chưa thành bệnh mãn tính, nhưng chỉ số của bạn đang ở ngưỡng cảnh báo (HbA1c: {hba1c}, Glucose: {glucose}). Cần điều chỉnh lối sống.")
        else:
            st.success("🟢 **AN TOÀN: BÌNH THƯỜNG**\nCác chỉ số của bạn nằm trong giới hạn an toàn.")
            if risk_score > 20:
                st.info("⚠️ Tuy nhiên, điểm rủi ro cao hơn mức trung bình, bạn vẫn nên duy trì chế độ ăn uống lành mạnh.")