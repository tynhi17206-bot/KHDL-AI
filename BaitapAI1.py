import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu
df = pd.read_csv("financial_statement.csv")

# Chuyển nhãn thành số
le = LabelEncoder()
df['Financial_Status'] = le.fit_transform(df['Financial_Status'])

# Tách dữ liệu
X = df.drop("Financial_Status", axis=1)
y = df["Financial_Status"]

# Chia train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Tạo model AI
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Lưu model
joblib.dump(model, "fraud_model.pkl")

# Lưu tên cột
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Train model thành công!")