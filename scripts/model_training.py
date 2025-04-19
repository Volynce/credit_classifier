import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from sklearn.impute import SimpleImputer

# Загрузка данных
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')

# Проверка на пропущенные значения в X_train и X_test
print("Количество пропущенных значений в X_train:\n", X_train.isnull().sum())
print("Количество пропущенных значений в X_test:\n", X_test.isnull().sum())

# Заполнение пропущенных значений средними значениями с помощью SimpleImputer
imputer = SimpleImputer(strategy='mean')

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Проверка на пропущенные значения после заполнения
print("Количество пропущенных значений в X_train после заполнения:\n", X_train_imputed.isnull().sum())
print("Количество пропущенных значений в X_test после заполнения:\n", X_test_imputed.isnull().sum())

# Преобразование категориальных данных в числовые (проверим, что в этих столбцах нет пропусков)
X_train_imputed['Пол'] = X_train_imputed['Пол'].map({'Мужской': 0, 'Женский': 1}).fillna(0).astype(int)
X_test_imputed['Пол'] = X_test_imputed['Пол'].map({'Мужской': 0, 'Женский': 1}).fillna(0).astype(int)

X_train_imputed['Состоит.в.браке'] = X_train_imputed['Состоит.в.браке'].map({'Не женат/не замужем': 0, 'Женат/замужем': 1}).fillna(0).astype(int)
X_test_imputed['Состоит.в.браке'] = X_test_imputed['Состоит.в.браке'].map({'Не женат/не замужем': 0, 'Женат/замужем': 1}).fillna(0).astype(int)

# Нормализация данных после заполнения пропусков
scaler = StandardScaler()

# Нормализуем только обучающие данные (fit_transform) и тестовые данные (transform)
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Проверка на NaN после нормализации
print("Количество пропущенных значений в X_train после нормализации:\n", pd.DataFrame(X_train_scaled).isnull().sum())
print("Количество пропущенных значений в X_test после нормализации:\n", pd.DataFrame(X_test_scaled).isnull().sum())

# Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train_scaled, y_train.values.ravel())  # Используем .values.ravel() для одномерного массива

# Оценка модели
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Сохранение модели
joblib.dump(model, '../models/logistic_regression_model.pkl')

print(f"Точность модели: {accuracy:.4f}")
