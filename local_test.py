import pandas as pd
import numpy as np
import joblib
import warnings

from processing import encoding_data

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings("ignore")

def load_assets():
    """Tải các file cần thiết: Model và Danh sách cột"""
    try:
        model = joblib.load('diabetes_model.joblib') 
        model_columns = joblib.load('model_columns.joblib') 
        print("✅ Đã tải thành công Model và danh sách Cột!")
        return model, model_columns
    except FileNotFoundError as e:
        print(f"❌ Lỗi: Không tìm thấy file. Hãy chắc chắn bạn đã chạy file train.py.\nChi tiết: {e}")
        return None, None


def get_user_input():
    """Hàm nhập liệu từ bàn phím"""
    print("\n--- NHẬP THÔNG TIN SỨC KHỎE ---")
    
    # 1. Nhập Gender
    print("1. Giới tính (Nhập: Male, Female, hoặc Other):")
    gender = input("   > ").strip()
    
    # 2. Nhập Age
    while True:
        try:
            age = float(input("2. Tuổi: "))
            break
        except ValueError: print("   Lỗi: Vui lòng nhập số.")

    # 3. Nhập Hypertension
    print("3. Có bị cao huyết áp không? (0: Không, 1: Có):")
    while True:
        try:
            hypertension = int(input("   > "))
            if hypertension in [0, 1]: break
            else: print("   Vui lòng chỉ nhập 0 hoặc 1.")
        except ValueError: print("   Lỗi: Nhập sai định dạng.")

    # 4. Nhập Heart Disease
    print("4. Có bị bệnh tim không? (0: Không, 1: Có):")
    while True:
        try:
            heart_disease = int(input("   > "))
            if heart_disease in [0, 1]: break
            else: print("   Vui lòng chỉ nhập 0 hoặc 1.")
        except ValueError: print("   Lỗi: Nhập sai định dạng.")

    # 5. Nhập Smoking History
    print("5. Lịch sử hút thuốc (Nhập: never, current, former, ever, not current, No Info):")
    smoking_history = input("   > ").strip()

    # 6. Nhập BMI
    while True:
        try:
            bmi = float(input("6. Chỉ số BMI (VD: 25.5): "))
            break
        except ValueError: print("   Lỗi: Vui lòng nhập số.")

    # 7. Nhập HbA1c
    while True:
        try:
            hbA1c = float(input("7. Chỉ số HbA1c (VD: 5.8): "))
            break
        except ValueError: print("   Lỗi: Vui lòng nhập số.")

    # 8. Nhập Blood Glucose
    while True:
        try:
            glucose = float(input("8. Chỉ số đường huyết (mg/dL) (VD: 140): "))
            break
        except ValueError: print("   Lỗi: Vui lòng nhập số.")

    # Tạo DataFrame từ dữ liệu nhập
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'hbA1c_level': hbA1c,
        'blood_glucose_level': glucose
    }])
    
    return input_data




def preprocess_input(input_df, model_columns):
    """
    Xử lý dữ liệu nhập vào cho khớp với định dạng lúc huấn luyện
    """
    input_df_encoded = encoding_data(input_df, categorical_cols=['gender', 'smoking_history'])

    input_df_ready = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    return input_df_ready




def main():
    # 1. Tải tài nguyên
    model, model_columns = load_assets()
    if model is None: return

    while True:
        # 2. Nhập liệu
        raw_input_df = get_user_input()
        
        # 3. Xử lý dữ liệu
        try:
            final_input = preprocess_input(raw_input_df, model_columns)
        except Exception as e:
            print(f"Lỗi xử lý dữ liệu: {e}")
            print("Mẹo: Kiểm tra lại xem tên các giá trị nhập (Male, never...) có đúng chính tả không.")
            continue

        # 4. Dự đoán
        prediction = model.predict(final_input)
        probability = model.predict_proba(final_input)

        # 5. Hiển thị kết quả
        print("\n" + "="*40)
        print("          KẾT QUẢ DỰ ĐOÁN")
        print("="*40)
        
        prob_percent = probability[0][1] * 100
        
        if prediction[0] == 1:
            print(f"🔴 CẢNH BÁO: Có nguy cơ mắc bệnh TIỂU ĐƯỜNG.")
            print(f"📊 Tỷ lệ rủi ro: {prob_percent:.2f}%")
        else:
            print(f"🟢 AN TOÀN: Ít nguy cơ mắc bệnh tiểu đường.")
            print(f"📊 Tỷ lệ rủi ro: {prob_percent:.2f}%")
        
        print("="*40 + "\n")
        
        cont = input("Bạn có muốn dự đoán cho người khác không? (y/n): ")
        if cont.lower() != 'y':
            break



if __name__ == "__main__":
    main()