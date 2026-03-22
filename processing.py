import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib


def encoding_data(df, categorical_cols):
    # # Mã hóa one-hot encoding cho các cột phân loại
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    if 'gender_Other' in df_encoded.columns:
        df_encoded = df_encoded.drop('gender_Other', axis=1)
        
    if 'smoking_history_ever' in df_encoded.columns:
        df_encoded = df_encoded.drop('smoking_history_ever', axis=1)

    return df_encoded



def data_preprocessing():
     
    # ------------------------
    #           EDA
    # ------------------------


    df_100k = pd.read_csv('diabetes_dataset_with_notes.csv')


    # ==========================
    # Kiểm tra dữ liệu
    # ==========================


    # print('---Bộ dữ liệu đầy đủ thuộc tính---\n')
    # print(df_100k.info())

    # print('-------------------------------\n')


    # print("--- 5 dòng đầu tiên ---")
    # print(df_100k.head())



    # ==========================
    # Xóa dữ liệu trùng lặp
    # ==========================


    duplicate_count = df_100k.duplicated().sum()

    # print('---Trước khi xóa dữ liệu trùng lặp---\n')
    # print(f"Số lượng dòng trùng lặp: {duplicate_count}\n")
    # print(f"Số lượng dòng dữ liệu: {df_100k_dropped.shape[0]}")

    # print('-------------------------------\n')

    # Xóa bỏ các dòng trùng lặp
    if duplicate_count > 0:
        # inplace=True: Thay đổi trực tiếp trên dataframe hiện tại
        df_100k.drop_duplicates(inplace=True)
        
        # reset_index: Đánh lại số thứ tự index cho đẹp (từ 0 đến n)
        df_100k.reset_index(drop=True, inplace=True)

    # # Recheck duplicate values
    # print('---Sau khi xóa dữ liệu trùng lặp---\n')
    # print(f"Số lượng dòng trùng lặp: {df_100k_dropped.duplicated().sum()}\n")
    # print(f"Số lượng dòng dữ liệu: {df_100k_dropped.shape[0]}")



    # print('-------------------------------\n')

    df_100k_dropped = df_100k.drop(columns=['year', 'location', 'race:AfricanAmerican', 
                                            'race:Asian', 'race:Caucasian', 'race:Hispanic', 
                                            'race:Other', 'clinical_notes'])


    # print('---Bộ dữ liệu đã lọc---\n')
    # print(df_100k_dropped.info())


    # print('-------------------------------\n')


    # ==========================
    # Vẽ biểu đồ
    # ==========================


    numerical_cols = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']

    # print('---Mô tả dữ liệu số học---\n')
    # print(numerical_cols.describe())


    # print('-------------------------------\n')

    # Vẽ Histogram (Xem phân phối)
    # plt.figure(figsize=(12, 8))
    # for i, col in enumerate(numerical_cols):
    #     plt.subplot(2, 2, i+1)
    #     sns.histplot(df_100k[col], kde=True, bins=30, color='blue')
    #     plt.title(f'Phân phối của {col}')
    # plt.tight_layout()
    # plt.show()


    # Vẽ Boxplot (Xem Outliers)
    # plt.figure(figsize=(12, 8))
    # for i, col in enumerate(numerical_cols):
    #     plt.subplot(2, 2, i+1)
    #     sns.boxplot(x=df_100k[col], color='orange')
    #     plt.title(f'Boxplot của {col}')
    # plt.tight_layout()
    # plt.show()


    categorical_cols = ['gender', 'smoking_history']


    # Vẽ biểu đồ phân tích phân bố của các cột phân loại
    # plt.figure(figsize=(11, 5))
    # for i, col in enumerate(categorical_cols):
    #     if col in df_100k.columns:
    #         plt.subplot(1, 2, i+1)
    #         sns.countplot(y=col, hue=col, data=df_100k, order=df_100k[col].value_counts().index, palette='pastel', legend=False)
    #         plt.title(f'Phân bố của {col}')
    # plt.tight_layout()
    # plt.show()



    # Vẽ biểu đồ phân tích biến mục tiêu
    # plt.figure(figsize=(6, 4))
    # sns.countplot(x='diabetes', hue='diabetes', data=df_100k, palette='viridis', legend=False)
    # plt.title('Phân bố của biến mục tiêu (Diabetes)')
    # plt.xlabel('0: Không bệnh - 1: Có bệnh')
    # plt.ylabel('Số lượng')
    # plt.show()




    # Biểu đồ phân tích tương quan 

    # Tạo một bản sao của dữ liệu để không làm ảnh hưởng đến dữ liệu gốc
    # df_heatmap = df_100k_dropped.copy()

    # df_heatmap = df_heatmap.rename(columns={'blood_glucose_level': 'blood_glucose'})


    # Khởi tạo bộ mã hóa
    # le = LabelEncoder()


    # Xác định các cột kiểu văn bản (object)
    # text_cols = df_heatmap.select_dtypes(include=['object']).columns
    # print(f"Các cột sẽ được mã hóa để vẽ Heatmap: {list(text_cols)}")


    # Thực hiện mã hóa (Label Encoding) cho từng cột
    # for col in text_cols:
    #     # Chuyển text thành số (0, 1, 2...)
    #     df_heatmap[col] = le.fit_transform(df_heatmap[col])
    #     # In ra để bạn hiểu nó đã mã hóa cái gì thành số mấy (tùy chọn)
    #     print(f" - Cột '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")


    # Tính toán ma trận tương quan
    # corr_matrix = df_heatmap.corr()


    # Vẽ Heatmap
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix, 
    #             annot=True,       # Hiển thị con số trong ô
    #             fmt=".2f",        # Làm tròn 2 chữ số thập phân
    #             cmap='coolwarm',  # Chọn tông màu (Xanh-Đỏ)
    #             linewidths=0.5)   # Đường viền giữa các ô
    # plt.title('Biểu đồ tương quan')
    # plt.xticks(rotation=15, ha='right')
    # plt.tight_layout()
    # plt.show()


    # ==========================
    # Mã hóa dữ liệu phân loại
    # ==========================


    # # Mã hóa one-hot encoding cho các cột phân loại
    # df_processed = encoding_data(df_100k_dropped, categorical_cols)

    df_processed = pd.get_dummies(df_100k_dropped, columns=categorical_cols, drop_first=False)
    if 'gender_Other' in df_processed.columns:
        df_processed = df_processed.drop('gender_Other', axis=1)
        
    if 'smoking_history_ever' in df_processed.columns:
        df_processed = df_processed.drop('smoking_history_ever', axis=1)

    # print("Các cột mới sau khi mã hóa:", df_processed.columns)

    # print('-------------------------------\n')

    # print(df_processed.info())

    # print('-------------------------------\n')


    return df_processed






