import tkinter as tk
from tkinter import ttk

class ModelSettingsWindow:
    def __init__(self, parent, callback):
        self.parent = parent
        self.callback = callback
        self.window = tk.Toplevel(parent)
        self.window.title("Настройки обучения модели")
        self.window.geometry("400x500")
        self.window.attributes('-topmost', True)

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        settings_frame = ttk.LabelFrame(main_frame, text="Параметры модели", padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Label(settings_frame, text="Количество деревьев:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.n_estimators = tk.IntVar(value=100)
        ttk.Entry(settings_frame, textvariable=self.n_estimators).grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(settings_frame, text="Скорость обучения:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.learning_rate = tk.DoubleVar(value=0.1)
        ttk.Entry(settings_frame, textvariable=self.learning_rate).grid(row=1, column=1, sticky=tk.EW, pady=2)

        ttk.Label(settings_frame, text="Максимальная глубина:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.max_depth = tk.IntVar(value=3)
        ttk.Entry(settings_frame, textvariable=self.max_depth).grid(row=2, column=1, sticky=tk.EW, pady=2)

        ttk.Label(settings_frame, text="Размер тестовой выборки (%):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.test_size = tk.IntVar(value=30)
        ttk.Scale(settings_frame, from_=10, to=40, variable=self.test_size, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=tk.EW, pady=2)
        ttk.Label(settings_frame, textvariable=self.test_size).grid(row=3, column=2, sticky=tk.W, padx=5)

        ttk.Label(settings_frame, text="Random state:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.random_state = tk.IntVar(value=42)
        ttk.Entry(settings_frame, textvariable=self.random_state).grid(row=4, column=1, sticky=tk.EW, pady=2)

        ttk.Label(settings_frame, text="Обработка дисбаланса:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.balance_classes = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.balance_classes).grid(row=5, column=1, sticky=tk.W, pady=2)

        ttk.Label(settings_frame, text="Удаление выбросов:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.remove_outliers = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.remove_outliers).grid(row=6, column=1, sticky=tk.W, pady=2)

        ttk.Label(settings_frame, text="Уровень загрязнения:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.contamination = tk.DoubleVar(value=0.05)
        ttk.Entry(settings_frame, textvariable=self.contamination).grid(row=7, column=1, sticky=tk.EW, pady=2)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Начать обучение", command=self.start_training).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Отмена", command=self.window.destroy).pack(side=tk.RIGHT, padx=5)

    def start_training(self):
        settings = {
            'n_estimators': self.n_estimators.get(),
            'learning_rate': self.learning_rate.get(),
            'max_depth': self.max_depth.get(),
            'test_size': self.test_size.get() / 100,
            'random_state': self.random_state.get(),
            'balance_classes': self.balance_classes.get(),
            'remove_outliers': self.remove_outliers.get(),
            'contamination': self.contamination.get()
        }
        self.window.destroy()
        self.callback(settings)
