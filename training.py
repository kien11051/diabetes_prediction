from processing import *
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE



# ===================================
# Clinical Diabetes Dataset
# ===================================


df = data_preprocessing()



# Chia dữ liệu thành tập đặc trưng và nhãn
x_processed = df.drop('diabetes', axis=1)
y_processed = df['diabetes']


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_processed_train, x_processed_test, y_processed_train, y_processed_test = train_test_split(x_processed, y_processed, 
                                                                                            test_size=0.2, stratify=y_processed, 
                                                                                            random_state=42)


# print('---Tập huấn luyện---\n')

# print(f"Số dòng dữ liệu: {x_processed_train.shape[0]}\n")
# print(f"Các thuộc tính:")
# print(x_processed_train.info())

# print('-------------------------------\n')

# print('---Tập kiểm tra---\n')

# print(f"Số dòng dữ liệu: {x_processed_test.shape[0]}\n")
# print(f"Các thuộc tính:")
# print(x_processed_test.info())

# print('-------------------------------\n')


# Cân bằng dữ liệu sử dụng SMOTE
smote = SMOTE(random_state=42)
x_processed_train_resampled, y_processed_train_resampled = smote.fit_resample(x_processed_train, y_processed_train)

# print('---Tập huấn luyện trước khi cân bằng---\n')

# print(f"Số dòng dữ liệu: {x_processed_train.shape[0]}\n")

# df_concat = pd.concat([x_processed_train, y_processed_train], axis=1)
# print(f"Tỉ lệ biến mục tiêu: {df_concat['diabetes'].value_counts(normalize=True) * 100}")

# print('-------------------------------\n')




# print('---Tập huấn luyện sau khi cân bằng---\n')

# print(f"Số dòng dữ liệu: {x_processed_train_resampled.shape[0]}\n")

# df_concat_smote = pd.concat([x_processed_train_resampled, y_processed_train_resampled], axis=1)
# print(f"Tỉ lệ biến mục tiêu: {df_concat_smote['diabetes'].value_counts(normalize=True) * 100}")




print('-------------------------------\n')



# ------------------------
#       Train_model
# ------------------------



classifier = RandomForestClassifier(random_state=42, n_estimators=500)



# classifier.fit(x_processed_train, y_processed_train)
# y_processed_pred = classifier.predict(x_processed_test)

# print('Mô hình huấn luyện với dữ liệu mất cân bằng\n')

# print(classification_report(y_processed_test, y_processed_pred))




# print('-------------------------------\n')




classifier.fit(x_processed_train_resampled, y_processed_train_resampled)
y_processed_pred_balance = classifier.predict(x_processed_test)

print('Mô hình huấn luyện với dữ liệu cân bằng\n')

print(classification_report(y_processed_test, y_processed_pred_balance))



print('-------------------------------\n')



# Lưu mô hình 
joblib.dump(classifier, 'diabetes_model.joblib')

# Lưu tên cột 
model_columns = x_processed_train.columns.tolist() 
joblib.dump(model_columns, 'model_columns.joblib')
print("Đã lưu mô hình và danh sách cột thành công!")



# ===========================
# Vẽ các biểu đồ phân tích
# ===========================

# Vẽ Ma trận nhầm lẫn (Confusion Matrix) unbalance
# cm = confusion_matrix(y_processed_test, y_processed_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
# plt.xlabel('Dự đoán (Predicted)', fontsize=14)
# plt.ylabel('Thực tế (Actual)', fontsize=14)
# plt.title('Ma trận nhầm lẫn (Dữ liệu mất cân bằng)', fontsize=16)
# plt.xticks([0.5, 1.5], ['Không bệnh', 'Có bệnh'], fontsize=12)
# plt.yticks([0.5, 1.5], ['Không bệnh', 'Có bệnh'], fontsize=12)
# plt.show()



# Vẽ Ma trận nhầm lẫn (Confusion Matrix) balance
# cm = confusion_matrix(y_processed_test, y_processed_pred_balance)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
# plt.xlabel('Dự đoán (Predicted)', fontsize=14)
# plt.ylabel('Thực tế (Actual)', fontsize=14)
# plt.title('Ma trận nhầm lẫn (Dữ liệu cân bằng)', fontsize=16)
# plt.xticks([0.5, 1.5], ['Không bệnh', 'Có bệnh'], fontsize=12)
# plt.yticks([0.5, 1.5], ['Không bệnh', 'Có bệnh'], fontsize=12)
# plt.show()




# Biểu đồ phân tích ảnh hưởng thuộc tính
# importances = classifier.feature_importances_
# feature_names = x_processed_train.columns 

# # Tạo dataframe để vẽ
# forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# # Vẽ biểu đồ
# plt.figure(figsize=(10, 6))
# forest_importances.plot.bar(color='skyblue')
# plt.title("Tầm quan trọng của các thuộc tính (Feature Importance)")
# plt.ylabel("Điểm quan trọng (Mean Decrease in Impurity)")
# plt.tight_layout()
# plt.show()





# # Biểu đồ ngưỡng
# # Lấy mẫu ngẫu nhiên (Subsampling)
# x_sample = x_processed_train_resampled.sample(n=5000, random_state=42)

# features_to_plot = ['hbA1c_level', 'blood_glucose_level', 'age', 'bmi']

# # Vẽ biểu đồ
# fig, ax = plt.subplots(figsize=(14, 10))

# display = PartialDependenceDisplay.from_estimator(
#     classifier, 
#     x_sample, 
#     features_to_plot,
#     kind="average",
#     n_cols=2, # Vẽ 2 cột
#     line_kw={"color": "red", "linewidth": 2},
#     ax=ax
# )

# plt.suptitle("Biểu đồ PDP: Ngưỡng rủi ro (Vẽ trên 5000 mẫu)", fontsize=16)
# plt.subplots_adjust(top=0.9, hspace=0.3)
# plt.show()