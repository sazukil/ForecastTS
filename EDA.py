import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('fraud_detection_dataset.csv', parse_dates=['timestamp'])

# 1. Структура данных
print('\nДатасет:', data.head(), sep='\n', end='\n')
print('\nИнформация о датасете:')
data.info()
print('\n\nПроверка наличия пропущенных значений:', data.isnull().sum(), sep='\n', end='\n')
print('\nАнализ целевой переменной:', data['is_fraud'].value_counts(normalize=True), sep='\n', end='\n')
print('\nОписательная статистика:', data.describe(), sep='\n', end='\n')


# 2. Распределение классов
ax = data['is_fraud'].value_counts().plot(kind='bar', title='Распределение классов (is_fraud)')
for i in ax.patches:
    ax.annotate(str(i.get_height()),
                (i.get_x() + i.get_width() / 2., i.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points')
plt.show()


# 3. Визуальный анализ временного ряда
#######
plt.figure(figsize=(18, 8))

daily_counts = data.set_index('timestamp').resample('D').size()
plt.plot(daily_counts.index, daily_counts/daily_counts.max(),
        color='royalblue', label='Все транзакции (норм.)')

daily_fraud = data[data['is_fraud']==1].set_index('timestamp').resample('D').size()
plt.plot(daily_fraud.index, daily_fraud/daily_fraud.max(),
        color='indianred', linestyle='--', label='Мошенничество (норм.)')

plt.title('Сравнение общего количества и мошеннических транзакций', pad=20)
plt.xlabel('Дата')
plt.ylabel('Нормированные значения')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

comparison_df = pd.DataFrame({
    'total': daily_counts,
    'fraud': daily_fraud
}).fillna(0)

correlation = comparison_df.corr().loc['total', 'fraud']
print(f"Коэффициент корреляции Пирсона: {correlation:.3f}")

#######
plt.figure(figsize=(18, 6))

daily_counts = data.set_index('timestamp').resample('D').size()

plt.plot(daily_counts.index, daily_counts.values,
            color='royalblue', linewidth=2)

plt.title('Общее количество транзакций по дням', pad=20)
plt.xlabel('Дата')
plt.ylabel('Количество транзакций')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#######
plt.figure(figsize=(18, 6))

daily_fraud = data[data['is_fraud']==1].set_index('timestamp').resample('D').size()

plt.plot(daily_fraud.index, daily_fraud.values,
            color='indianred', linewidth=2)

plt.title('Количество мошеннических транзакций по дням', pad=20)
plt.xlabel('Дата')
plt.ylabel('Количество мошеннических операций')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.figure(figsize=(18, 6))

#######
plt.figure(figsize=(18, 6))

data['hour'] = data['timestamp'].dt.hour

hourly = data.groupby(['hour', 'is_fraud']).size().unstack()
hourly.columns = ['Легальные', 'Мошеннические']

hourly_norm = hourly.div(hourly.sum(axis=1), axis=0)

hourly_norm.plot(kind='bar', stacked=True,
                color=['lightsteelblue', 'lightcoral'],
                edgecolor='black', linewidth=0.5)

plt.title('Распределение транзакций по часам суток', pad=20)
plt.xlabel('Час дня')
plt.ylabel('Доля транзакций')
plt.xticks(rotation=0)
plt.grid(alpha=0.3, axis='y')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

#######
plt.figure(figsize=(18, 6))

data['day_of_week'] = data['timestamp'].dt.dayofweek
days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']

weekly = data.groupby(['day_of_week', 'is_fraud']).size().unstack()
weekly.index = days
weekly.columns = ['Легальные', 'Мошеннические']

weekly_norm = weekly.div(weekly.sum(axis=1), axis=0)

weekly_norm.plot(kind='bar', stacked=True,
                color=['lightsteelblue', 'lightcoral'],
                edgecolor='black', linewidth=0.5)

plt.title('Распределение транзакций по дням недели', pad=20)
plt.xlabel('День недели')
plt.ylabel('Доля транзакций')
plt.grid(alpha=0.3, axis='y')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

#######
plt.figure(figsize=(18, 6))

fraud_daily = data[data['is_fraud']==1].set_index('timestamp').resample('D').size()

rolling_mean = fraud_daily.rolling(window=window).mean()

plt.plot(fraud_daily.index, fraud_daily.values,
            color='lightcoral', alpha=0.5, label='Ежедневные значения')
plt.plot(rolling_mean.index, rolling_mean.values,
            color='darkred', linewidth=2,
            label=f'{window}-дневное скользящее среднее')

plt.title(f'Динамика мошеннических транзакций ({window}-дневное скользящее среднее)', pad=20)
plt.xlabel('Дата')
plt.ylabel('Количество мошеннических операций')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#######
plt.figure(figsize=(18, 6))

data['month_year'] = data['timestamp'].dt.to_period('M')

monthly_amounts = data.groupby(['month_year', 'is_fraud'])['amount'].mean().unstack()
monthly_amounts.columns = ['Легальные транзакции', 'Мошеннические транзакции']

monthly_amounts.plot(color=['dodgerblue', 'crimson'], linewidth=2.5)

plt.title('Средняя сумма транзакций по месяцам', pad=20)
plt.xlabel('Месяц')
plt.ylabel('Средняя сумма транзакции')
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

#######
numeric_cols = ['amount', 'age', 'income', 'debt', 'credit_score', 'is_fraud']
corr_data = data[numeric_cols]

corr_matrix = corr_data.corr(method='pearson')

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

heatmap = sns.heatmap(corr_matrix,
                      mask=mask,
                      cmap=cmap,
                      vmin=-1, vmax=1,
                      center=0,
                      annot=True,
                      fmt=".6f",
                      linewidths=.5,
                      cbar_kws={"shrink": .75})

plt.title('Матрица корреляций числовых признаков', pad=20, fontsize=16)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()


# 4. Анализ выбросов
numeric_cols = ['amount', 'age', 'income', 'debt', 'credit_score']
for col in numeric_cols:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=data[col])
    plt.title(f'Распределение {col} с выбросами')
    plt.show()
