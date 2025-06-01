import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import joblib


class FraudPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ кредитных транзакций")

        try:
            self.model = joblib.load('fraud_detection_model.pkl')
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {str(e)}")
            self.root.destroy()
            return

        self.create_widgets()
        self.transaction_data = None

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = ttk.LabelFrame(main_frame, text="Загрузка данных", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Обзор...", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Анализировать", command=self.analyze_data).pack(side=tk.RIGHT, padx=5)

        result_frame = ttk.LabelFrame(main_frame, text="Результаты анализа", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = [
            'ID', 'Сумма', 'Местоположение', 'Тип устройства', 'Возраст',
            'Доход', 'Долг', 'Рейтинг', 'Прогноз'
        ]
        self.tree = ttk.Treeview(result_frame, columns=columns, show='headings')

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor=tk.CENTER)

        self.tree.column('ID', width=50, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.stats_label = ttk.Label(main_frame, text="", style='Header.TLabel')
        self.stats_label.pack(pady=5)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Очистить", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сохранить результаты", command=self.save_results).pack(side=tk.LEFT, padx=5)

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.file_path.set(filepath)

    def analyze_data(self):
        filepath = self.file_path.get()
        if not filepath:
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите файл для анализа")
            return

        try:
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
            else:
                data = pd.read_excel(filepath)

            required_columns = [
                'timestamp', 'amount', 'location',
                'device_type', 'age', 'income',
                'debt', 'credit_score'
            ]

            if not all(col in data.columns for col in required_columns):
                missing = set(required_columns) - set(data.columns)
                messagebox.showerror("Ошибка", f"В файле отсутствуют необходимые столбцы: {missing}")
                return

            self.clear_results()
            self.transaction_data = data
            fraud_count = 0

            for i, row in data.iterrows():
                transaction = row.to_dict()
                prediction = self.predict_transaction(transaction)

                is_fraud = prediction == 1
                if is_fraud:
                    fraud_count += 1

                self.tree.insert('', 'end', values=(
                    i+1,
                    f"{transaction['amount']:.2f}",
                    transaction['location'],
                    transaction['device_type'],
                    transaction['age'],
                    transaction['income'],
                    transaction['debt'],
                    transaction['credit_score'],
                    "Мошенничество" if prediction == 1 else "Легальная"
                ))

                if is_fraud:
                    self.tree.tag_configure('fraud', background='#ffcccc')
                    self.tree.item(self.tree.get_children()[-1], tags=('fraud',))

            total = len(data)
            self.stats_label.config(
                text=f"Проанализировано транзакций: {total} | Подозрительных: {fraud_count} ({fraud_count/total:.2%})"
            )

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при анализе файла: {str(e)}")

    def predict_transaction(self, transaction):
        input_df = pd.DataFrame([transaction])

        if 'timestamp' in input_df.columns:
            input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
            input_df['hour'] = input_df['timestamp'].dt.hour
            input_df['day_of_week'] = input_df['timestamp'].dt.dayofweek
            input_df['day_of_month'] = input_df['timestamp'].dt.day
            input_df['month'] = input_df['timestamp'].dt.month

        cols_to_drop = ['timestamp', 'user_id']
        for col in cols_to_drop:
            if col in input_df.columns:
                input_df = input_df.drop(col, axis=1)

        prediction = self.model.predict(input_df)[0]

        return prediction

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.stats_label.config(text="")
        self.transaction_data = None

    def save_results(self):
        if not self.tree.get_children():
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )

        if not filepath:
            return

        try:
            data = []
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                data.append({
                    'ID': values[0],
                    'Сумма': values[1],
                    'Местоположение': values[2],
                    'Тип устройства': values[3],
                    'Возраст': values[4],
                    'Доход': values[5],
                    'Долг': values[6],
                    'Кредитный рейтинг': values[7],
                    'Прогноз': values[8]
                })

            df = pd.DataFrame(data)
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)

            messagebox.showinfo("Успех", "Результаты успешно сохранены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FraudPredictorApp(root)
    root.mainloop()
