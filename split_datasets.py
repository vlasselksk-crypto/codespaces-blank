import pandas as pd
import os
from pathlib import Path

# Создаем папки
ru_data_path = Path('/workspaces/codespaces-blank/ru_data')
legit_dir = ru_data_path / 'legit'
cheat_dir = ru_data_path / 'cheat'

legit_dir.mkdir(exist_ok=True)
cheat_dir.mkdir(exist_ok=True)

# Файлы для обработки
dataset_files = ['dataset.csv', 'dataset2.csv', 'dataset3.csv']

print("Разделение данных на LEGIT и CHEAT...")

for filename in dataset_files:
    filepath = ru_data_path / filename
    
    try:
        # Читаем CSV с пропуском ошибочных строк
        df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
        
        # Проверяем что последняя колонка это label
        if df.columns[-1] != 'label':
            print(f"⚠️ {filename}: последняя колонка не 'label'")
            continue
        
        # Разделяем по label
        legit = df[df['label'] == 0]
        cheat = df[df['label'] == 1]
        
        # Сохраняем LEGIT
        legit_output = legit_dir / f'LEGIT#{filename.replace(".csv", "").replace("dataset", "dataset_split")}_legit.csv'
        legit.to_csv(legit_output, index=False)
        print(f"✓ {legit_output.name}: {len(legit)} строк")
        
        # Сохраняем CHEAT
        cheat_output = cheat_dir / f'CHEAT#{filename.replace(".csv", "").replace("dataset", "dataset_split")}_cheat.csv'
        cheat.to_csv(cheat_output, index=False)
        print(f"✓ {cheat_output.name}: {len(cheat)} строк")
        
    except Exception as e:
        print(f"✗ Ошибка при обработке {filename}: {e}")

print("\n✓ Разделение завершено!")

# Статистика
print("\nОбщая статистика:")
legit_files = list(legit_dir.glob('*.csv'))
cheat_files = list(cheat_dir.glob('*.csv'))
print(f"LEGIT файлов: {len(legit_files)}")
print(f"CHEAT файлов: {len(cheat_files)}")

# Объединяем все в один файл для обучения
print("\nОбъединение всех данных за каждый класс...")

legit_dfs = []
cheat_dfs = []

for f in legit_files:
    legit_dfs.append(pd.read_csv(f))

for f in cheat_files:
    cheat_dfs.append(pd.read_csv(f))

if legit_dfs:
    legit_combined = pd.concat(legit_dfs, ignore_index=True)
    legit_combined.to_csv(legit_dir / 'LEGIT_combined.csv', index=False)
    print(f"✓ LEGIT_combined.csv: {len(legit_combined)} строк")

if cheat_dfs:
    cheat_combined = pd.concat(cheat_dfs, ignore_index=True)
    cheat_combined.to_csv(cheat_dir / 'CHEAT_combined.csv', index=False)
    print(f"✓ CHEAT_combined.csv: {len(cheat_combined)} строк")
