import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import joblib
import time
import threading
from model_trainer import ModelTrainerWindow

class FraudPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ кредитных транзакций")
        self.root.geometry("1220x550")
        self.root.minsize(1220, 550)

        self.analysis_active = False
        self.should_stop = False
        self.start_time = 0

        self.model_path = 'models\default.pkl'
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель по умолчанию: {str(e)}")
            self.root.destroy()
            return

        self.create_widgets()
        self.transaction_data = None

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        model_frame = ttk.LabelFrame(main_frame, text="Управление моделью", padding=10)
        model_frame.pack(fill=tk.X, pady=5)

        self.model_label = ttk.Label(model_frame, text=f"Текущая модель: {self.model_path}")
        self.model_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(model_frame, text="Обучить модель", command=self.open_trainer_window).pack(side=tk.RIGHT, padx=5)
        ttk.Button(model_frame, text="Загрузить другую модель", command=self.load_model).pack(side=tk.RIGHT, padx=5)

        file_frame = ttk.LabelFrame(main_frame, text="Загрузка данных", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Обзор...", command=self.browse_file).pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(file_frame)
        action_frame.pack(side=tk.RIGHT, padx=5)

        self.analyze_button = ttk.Button(action_frame, text="Анализировать", command=self.start_analysis_thread)
        self.analyze_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = ttk.Button(action_frame, text="Остановить", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)

        result_frame = ttk.LabelFrame(main_frame, text="Результаты анализа", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = [
            'ID', 'Сумма', 'Местоположение', 'Тип устройства', 'Возраст',
            'Доход', 'Долг', 'Рейтинг', 'Прогноз', 'Вероятность'
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

        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="", style='Header.TLabel')
        self.stats_label.pack(side=tk.LEFT, padx=5)

        self.time_label = ttk.Label(stats_frame, text="Время анализа: 00:00:00")
        self.time_label.pack(side=tk.RIGHT, padx=5)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Очистить", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сохранить результаты", command=self.save_results).pack(side=tk.LEFT, padx=5)

    def open_trainer_window(self):
        ModelTrainerWindow(self.root)

    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Model files", "*.pkl")]
        )
        if filepath:
            try:
                self.model = joblib.load(filepath)
                self.model_path = filepath
                self.model_label.config(text=f"Текущая модель: {filepath}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {str(e)}")

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.file_path.set(filepath)

    def start_analysis_thread(self):
        if self.analysis_active:
            return

        filepath = self.file_path.get()
        if not filepath:
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите файл для анализа")
            return

        self.should_stop = False
        self.analysis_active = True
        self.analyze_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.clear_results()
        self.start_time = time.time()
        self.update_timer()
        threading.Thread(target=self.analyze_data, args=(filepath,), daemon=True).start()

    def stop_analysis(self):
        self.should_stop = True
        self.stop_button.config(state=tk.DISABLED)

    def update_timer(self):
        if self.analysis_active:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_label.config(text=f"Время анализа: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            self.root.after(1000, self.update_timer)

    def analyze_data(self, filepath):
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
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"В файле отсутствуют необходимые столбцы: {missing}"))
                return

            self.transaction_data = data
            fraud_count = 0
            processed_count = 0

            for i, row in data.iterrows():
                if self.should_stop:
                    break

                transaction = row.to_dict()
                prediction, probability = self.predict_transaction(transaction)

                is_fraud = prediction == 1
                if is_fraud:
                    fraud_count += 1

                self.root.after(0, self.add_tree_item, i, transaction, prediction, probability, is_fraud)
                processed_count += 1

            total = len(data)
            fraud_percent = "0.00%"
            if processed_count > 0:
                fraud_percent = f"{fraud_count/processed_count:.2%}"
            self.root.after(0, lambda: self.stats_label.config(
                text=f"Обработано транзакций: {processed_count}/{total} | Подозрительных: {fraud_count} ({fraud_percent})"
            ))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Произошла ошибка при анализе файла: {str(e)}"))
        finally:
            self.analysis_active = False
            self.should_stop = False
            self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))

    def add_tree_item(self, i, transaction, prediction, probability, is_fraud):
        item = self.tree.insert('', 'end', values=(
            i+1,
            f"{transaction['amount']:.2f}",
            transaction['location'],
            transaction['device_type'],
            transaction['age'],
            transaction['income'],
            transaction['debt'],
            transaction['credit_score'],
            "Мошенничество" if prediction == 1 else "Легальная",
            f"{probability:.4f}"
        ))

        if is_fraud:
            self.tree.tag_configure('fraud', background='#ffcccc')
            self.tree.item(item, tags=('fraud',))

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
        probability = self.model.predict_proba(input_df)[0][1]

        return prediction, probability

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
            filetypes=[("CSV files", "*.csv")]
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
                    'Прогноз': values[8],
                    'Вероятность': values[9]
                })

            df = pd.DataFrame(data)
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)

            messagebox.showinfo("Успех", "Результаты успешно сохранены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
