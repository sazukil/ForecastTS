import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import joblib
import time
import threading
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, accuracy_score,
                           precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from model_settings import ModelSettingsWindow

class ModelTrainerWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Обучение модели")
        self.window.geometry("800x600")
        self.window.attributes('-topmost', True)

        self.training_data = None
        self.training_active = False
        self.start_time = 0
        self.timer_id = None
        self.model = None
        self.training_complete = False
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = ttk.LabelFrame(main_frame, text="Загрузка данных", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Обзор...", command=self.browse_file).pack(side=tk.LEFT, padx=5)

        action_frame = ttk.Frame(file_frame)
        action_frame.pack(side=tk.RIGHT, padx=5)

        self.load_button = ttk.Button(action_frame, text="Загрузить данные", command=self.load_training_data)
        self.load_button.pack(side=tk.LEFT, padx=2)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        metrics_frame = ttk.LabelFrame(content_frame, text="Метрики модели", padding=10)
        metrics_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.timer_label = ttk.Label(metrics_frame, text="Время обучения: 0:00:00")
        self.timer_label.pack(pady=5)

        self.start_button = ttk.Button(metrics_frame, text="Начать обучение", command=self.start_training)
        self.start_button.pack(pady=5)

        self.progress_label = ttk.Label(metrics_frame, text="Статус: Ожидание данных")
        self.progress_label.pack(pady=5)

        self.metrics_text = tk.Text(metrics_frame, height=15, state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = ttk.Scrollbar(self.metrics_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.metrics_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.metrics_text.yview)

        self.confusion_frame = ttk.LabelFrame(content_frame, text="Матрица ошибок", padding=10)
        self.confusion_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.confusion_placeholder = ttk.Label(self.confusion_frame, text="Матрица ошибок появится после обучения")
        self.confusion_placeholder.pack(expand=True)

        self.save_model_button = ttk.Button(main_frame, text="Сохранить модель", command=self.save_model)
        self.save_model_button.pack(pady=10)

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.file_path.set(filepath)
            self.progress_label.config(text="Статус: Данные выбраны, готово к загрузке")

    def load_training_data(self):
        filepath = self.file_path.get()
        if not filepath:
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите файл с данными")
            return

        try:
            self.training_data = pd.read_csv(filepath)
            required_columns = [
                'timestamp', 'amount', 'location', 'device_type',
                'age', 'income', 'debt', 'credit_score', 'is_fraud'
            ]

            if not all(col in self.training_data.columns for col in required_columns):
                missing = set(required_columns) - set(self.training_data.columns)
                messagebox.showerror("Ошибка", f"В файле отсутствуют необходимые столбцы: {missing}")
                return

            self.progress_label.config(text=f"Статус: Данные загружены успешно ({len(self.training_data)} записей)")
            self.start_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {str(e)}")
            self.progress_label.config(text="Статус: Ошибка загрузки данных")

    def start_training(self):
        if self.training_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные для обучения")
            return

        ModelSettingsWindow(self.window, self.start_training_with_settings)

    def start_training_with_settings(self, settings):
        if self.training_active:
            return

        self.training_active = True
        self.start_button.config(state=tk.DISABLED)
        self.progress_label.config(text="Статус: Обучение начато...")

        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "Метрики модели:\n...\n")
        self.metrics_text.config(state=tk.DISABLED)

        self.start_time = time.time()
        self.update_timer()

        threading.Thread(target=self.train_model, args=(settings,), daemon=True).start()

    def update_timer(self):
        if self.training_active:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.timer_label.config(text=f"Время обучения: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            self.timer_id = self.window.after(1000, self.update_timer)

    def train_model(self, settings):
        try:
            df = self.training_data.copy()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month

            X = df.drop(['is_fraud', 'timestamp', 'user_id'], axis=1)
            y = df['is_fraud']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=settings['test_size'],
                random_state=settings['random_state'],
                stratify=y
            )

            if settings['remove_outliers']:
                numeric_features = [
                    'amount', 'age', 'income', 'debt', 'credit_score',
                    'hour', 'day_of_week', 'day_of_month', 'month'
                ]

                outlier_detector = IsolationForest(
                    contamination=settings['contamination'],
                    random_state=settings['random_state']
                )
                outliers = outlier_detector.fit_predict(X_train[numeric_features])
                X_train = X_train[np.where(outliers != -1, True, False)]
                y_train = y_train[np.where(outliers != -1, True, False)]

            categorical_features = ['location', 'device_type']
            numeric_features = [
                'amount', 'age', 'income', 'debt', 'credit_score',
                'hour', 'day_of_week', 'day_of_month', 'month'
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])

            if settings['balance_classes']:
                model = make_pipeline(
                    preprocessor,
                    SMOTE(random_state=settings['random_state']),
                    GradientBoostingClassifier(
                        n_estimators=settings['n_estimators'],
                        learning_rate=settings['learning_rate'],
                        max_depth=settings['max_depth'],
                        random_state=settings['random_state']
                    )
                )
            else:
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', GradientBoostingClassifier(
                        n_estimators=settings['n_estimators'],
                        learning_rate=settings['learning_rate'],
                        max_depth=settings['max_depth'],
                        random_state=settings['random_state']
                    ))
                ])

            self.window.after(0, lambda: self.progress_label.config(
                text="Статус: Идет обучение модели..."
            ))

            model.fit(X_train, y_train)
            self.model = model

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)

            self.window.after(0, lambda: self.update_metrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                roc_auc=roc_auc,
                cm=cm
            ))

            self.window.after(0, lambda: self.progress_label.config(
                text="Статус: Обучение завершено!"
            ))

        except Exception as e:
            error_message = str(e)
            self.window.after(0, lambda: self.progress_label.config(
                text=f"Статус: Ошибка при обучении - {error_message}"
            ))
        finally:
            self.training_active = False
            self.window.after(0, lambda: self.start_button.config(state=tk.NORMAL))

            if self.timer_id:
                self.window.after_cancel(self.timer_id)

    def update_metrics(self, accuracy, precision, recall, f1, roc_auc, cm):
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)

        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        metrics_text = f"""Метрики модели:

Общее время обучения: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F-score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}
"""
        self.metrics_text.insert(tk.END, metrics_text)
        self.metrics_text.config(state=tk.DISABLED)

        for widget in self.confusion_frame.winfo_children():
            widget.destroy()

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Легальная', 'Мошенническая'],
                    yticklabels=['Легальная', 'Мошенническая'])
        ax.set_xlabel('Предсказанный')
        ax.set_ylabel('Фактический')
        ax.set_title('Матрица ошибок')

        canvas = FigureCanvasTkAgg(fig, master=self.confusion_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.confusion_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig = fig
        self.canvas = canvas
        self.toolbar = toolbar

    def save_model(self):
        if not self.model:
            messagebox.showwarning("Предупреждение", "Нет обученной модели для сохранения")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Model files", "*.pkl")]
        )

        if filepath:
            try:
                joblib.dump(self.model, filepath)

                self.metrics_text.config(state=tk.NORMAL)
                self.metrics_text.insert(tk.END, f"\n\nМодель успешно сохранена в: {filepath}")
                self.metrics_text.see(tk.END)
                self.metrics_text.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить модель: {str(e)}")
