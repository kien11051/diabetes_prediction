# 🩺 Dự đoán rủi ro bệnh Tiểu đường (Diabetes Prediction)

Dự án học máy (Machine Learning) áp dụng thuật toán Random Forest để dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số y khoa và nhân khẩu học.

## 🌟 Điểm nổi bật của dự án
* **Xử lý dữ liệu mất cân bằng:** Áp dụng kỹ thuật SMOTE để cải thiện khả năng dự đoán lớp thiểu số (người mắc bệnh).
* **Explainable AI (XAI):** Sử dụng Feature Importance và biểu đồ Partial Dependence Plots (PDP) để giải thích logic của mô hình, đối chiếu với tiêu chuẩn y khoa thực tế (ngưỡng Glucose >= 200).
* **Logic Y khoa tích hợp:** Kết hợp quy tắc chẩn đoán (Heuristics) để nhận diện các vùng xám (Tiền tiểu đường) mà mô hình thuần túy có thể bỏ sót.

## 🛠️ Công nghệ sử dụng
* Python, Pandas, Scikit-learn, Imbalanced-learn.
* Giao diện Web App: Streamlit.

## 🚀 Cách chạy dự án
1. Cài đặt các thư viện cần thiết: `pip install streamlit` 
2. Chạy ứng dụng web: `streamlit run app_web.py`