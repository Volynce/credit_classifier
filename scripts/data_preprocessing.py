import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных с правильным разделителем и без кавычек
data = pd.read_csv('../data/credit.txt', sep=';', quotechar='"', encoding='utf-8', engine='python')

# Очистка пробелов в названиях столбцов
data.columns = data.columns.str.strip()

# Проверка уникальных значений в столбцах
print("Уникальные значения в 'Пол':", data['Пол'].unique())
print("Уникальные значения в 'Состоит.в.браке':", data['Состоит.в.браке'].unique())

# Заполнение пропусков в "Пол" и "Состоит.в.браке" на 0 (если NaN)
data['Пол'] = pd.to_numeric(data['Пол'], errors='coerce').fillna(0).astype(int)
data['Состоит.в.браке'] = pd.to_numeric(data['Состоит.в.браке'], errors='coerce').fillna(0).astype(int)

# Проверка на пропущенные значения
print("Количество пропущенных значений:\n", data.isnull().sum())

# Заполнение оставшихся пропущенных значений средним для всех столбцов
data.fillna(data.mean(), inplace=True)

# Предварительный анализ данных
print(data.describe())
print(data.info())

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(columns=['Благонадежный.заемщик'])
y = data['Благонадежный.заемщик']

# Разбиение данных на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сохранение подготовленных данных для дальнейшего использования
X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)
