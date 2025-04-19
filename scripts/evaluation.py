import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Загрузка данных и модели
X_test = pd.read_csv('../data/X_test.csv')
y_test = pd.read_csv('../data/y_test.csv')

# Загрузка обучающих данных, чтобы гарантировать одинаковые столбцы
X_train = pd.read_csv('../data/X_train.csv')

# Проверка на пропущенные значения в X_test
print("Количество пропущенных значений в X_test:\n", X_test.isnull().sum())

# Заполнение пропущенных значений средними значениями
imputer = SimpleImputer(strategy='mean')
X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)

# Проверка на пропущенные значения после заполнения
print("Количество пропущенных значений в X_test после заполнения:\n", X_test_imputed.isnull().sum())

# Преобразование категориальных данных в числовые
X_test_imputed['Пол'] = X_test_imputed['Пол'].map({'Мужской': 0, 'Женский': 1}).fillna(0).astype(int)
X_test_imputed['Состоит.в.браке'] = X_test_imputed['Состоит.в.браке'].map({'Не женат/не замужем': 0, 'Женат/замужем': 1}).fillna(0).astype(int)

# Применение одинаковых признаков, как и в обучающем наборе
X_test_imputed = X_test_imputed[X_train.columns]  # Сохраняем все столбцы, которые есть в обучении

# Нормализация данных после заполнения пропусков
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_imputed)

# Проверка на NaN после нормализации
print("Количество пропущенных значений в X_test после нормализации:\n", pd.DataFrame(X_test_scaled).isnull().sum())

# Заполнение NaN после нормализации
X_test_scaled = imputer.transform(X_test_scaled)

# Проверка на NaN после обработки с помощью Imputer
print("Количество пропущенных значений в X_test после Imputer:\n", pd.DataFrame(X_test_scaled).isnull().sum())

# Загрузка модели
model = joblib.load('../models/logistic_regression_model.pkl')

# Предсказания
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Кросс-таблица
conf_matrix = confusion_matrix(y_test, y_pred)
print("Кросс-таблица:")
print(conf_matrix)

# Отчет о классификации
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# ROC-анализ
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Визуализация ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
