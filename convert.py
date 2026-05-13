import pandas as pd

# Читаем файл с заголовком
df = pd.read_csv("ru_data/dataset.csv", header=0, on_bad_lines='skip')

# Переименовываем колонки
df.rename(columns={
    'deltaYaw': 'delta_yaw',
    'deltaPitch': 'delta_pitch',
    'accelYaw': 'accel_yaw',
    'accelPitch': 'accel_pitch',
    'jerkYaw': 'jerk_yaw',
    'jerkPitch': 'jerk_pitch',
    'label': 'is_cheating'
}, inplace=True)

# Добавляем недостающие колонки
df['gcd_error_yaw'] = 0.0
df['gcd_error_pitch'] = 0.0

# Переставляем в нужный порядок
df = df[['is_cheating', 'delta_yaw', 'delta_pitch', 'accel_yaw', 'accel_pitch', 'jerk_yaw', 'jerk_pitch', 'gcd_error_yaw', 'gcd_error_pitch']]

# Сохраняем
df.to_csv("ru_data/dataset_converted.csv", index=False)
print("✅ Файл сконвертирован: ru_data/dataset_converted.csv")