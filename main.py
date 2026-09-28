# switch to plaidml-keras (optimise for university PCs)
import collections
import sys
if sys.version_info >= (3, 10):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# ---------- Датасет: локальный кеш, чтобы не дёргать kagglehub каждый раз ----------
def load_dataset():
    local_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Iris.csv")
    if not os.path.exists(local_csv):
        import kagglehub
        path = kagglehub.dataset_download("uciml/iris")
        pd.read_csv(os.path.join(path, "Iris.csv")).to_csv(local_csv, index=False)
    return pd.read_csv(local_csv)


def build_model(hidden, activation, optimizer):
    m = Sequential()
    m.add(Dense(hidden, input_dim=4, activation=activation))
    m.add(Dense(3, activation="softmax"))
    m.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    return m

class LivePlotCallback(Callback):
    """
    Keras calls on_epoch_end automatically
    """
    def __init__(self, app, line_acc, line_loss):
        super().__init__()
        self.app = app
        self.line_acc = line_acc
        self.line_loss = line_loss
        self.xs = []
        self.acc = []
        self.loss = []
        self.redraw_every = 5

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_acc = logs.get("val_accuracy")
        val_loss = logs.get("val_loss")

        if val_acc is None or val_loss is None:
            return

        self.xs.append(epoch + 1)
        self.acc.append(val_acc)
        self.loss.append(val_loss)


        self.line_acc.set_data(self.xs, self.acc)
        self.line_loss.set_data(self.xs, self.loss)

        if (epoch + 1) % self.redraw_every == 0:
            self.app.ax_acc.relim()
            self.app.ax_acc.autoscale_view()
            self.app.ax_loss.relim()
            self.app.ax_loss.autoscale_view()
            self.app.canvas.draw_idle()
            self.app.root.update_idletasks()

class App:
    def __init__(self, root):
        self.root = root
        root.title("Iris NN Lab")

        # ---------- данные ----------
        df = load_dataset()
        data = df.values
        X = data[:, 1:5].astype(float)
        Y = data[:, 5]
        self.encoder = LabelEncoder().fit(Y)
        self.X = X
        self.encoded_Y = self.encoder.transform(Y)
        self.dummy_y = to_categorical(self.encoded_Y)

        # ---------- GUI ----------
        frm = ttk.Frame(root, padding=8)
        frm.pack(fill="both", expand=True)

        # общие параметры
        params = ttk.LabelFrame(frm, text="Общие параметры", padding=6)
        params.pack(fill="x")
        self.test_size    = self._field(params, "test_size",    "0.2", 0)
        self.val_size     = self._field(params, "val_size",     "0.2", 1)
        self.random_state = self._field(params, "random_state", "42",  2)

        # конфиги
        cfg_box = ttk.LabelFrame(
            frm,
            text="Конфигурации (одна на строку: hidden, activation, optimizer, batch_size, epochs)",
            padding=6
        )
        cfg_box.pack(fill="x")
        self.cfg_text = scrolledtext.ScrolledText(cfg_box, height=6, font=("Consolas", 10))
        self.cfg_text.pack(fill="both", expand=True)
        self.cfg_text.insert("1.0",
            "8, relu, adam, 32, 200\n"
            "16, relu, adam, 16, 200\n"
            "8, tanh, rmsprop, 32, 200\n"
            "4, relu, sgd, 32, 200\n"
        )

        # кнопка запуска
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Запустить", command=self.run).pack(side="left")
        self.status = ttk.Label(btns, text="Готово")
        self.status.pack(side="left", padx=10)

        # лог
        log_box = ttk.LabelFrame(frm, text="Результаты", padding=6)
        log_box.pack(fill="both", expand=True)
        self.log = scrolledtext.ScrolledText(log_box, height=10, font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)

        # графики
        plot_box = ttk.LabelFrame(frm, text="Графики обучения (красный — лучшая модель по val_accuracy)",
                                  padding=6)
        plot_box.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(10, 4))
        self.ax_acc  = self.fig.add_subplot(1, 2, 1)
        self.ax_loss = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _field(self, parent, label, default, col):
        ttk.Label(parent, text=label).grid(row=0, column=col * 2, padx=4, sticky="e")
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=8).grid(row=0, column=col * 2 + 1, padx=4, sticky="w")
        return var

    def parse_configs(self):
        configs = []
        for line in self.cfg_text.get("1.0", "end").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                raise ValueError(f"Плохая строка: {line!r}")
            hidden, activation, optimizer, batch_size, epochs = parts
            configs.append({
                "hidden": int(hidden),
                "activation": activation,
                "optimizer": optimizer,
                "batch_size": int(batch_size),
                "epochs": int(epochs),
            })
        if not configs:
            raise ValueError("Не задано ни одной конфигурации")
        return configs

    def log_write(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.root.update_idletasks()

    # ---------- основной прогон ----------
    def run(self):
        try:
            configs = self.parse_configs()
            test_size    = float(self.test_size.get())
            val_size     = float(self.val_size.get())
            random_state = int(self.random_state.get())
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        # стратифицированное разбиение
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.dummy_y,
            test_size=test_size, random_state=random_state,
            stratify=self.encoded_Y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size, random_state=random_state,
            stratify=np.argmax(y_train, axis=1)
        )

        self.log.delete("1.0", "end")
        histories = []
        models = []
        results = []

        for i, cfg in enumerate(configs):
            self.status.config(text=f"Обучение {i + 1}/{len(configs)}...")
            self.root.update()
            m = build_model(cfg["hidden"], cfg["activation"], cfg["optimizer"])
            h = m.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                verbose=0
            )
            best_val = max(h.history["val_accuracy"])
            test_loss, test_acc = m.evaluate(X_test, y_test, verbose=0)

            results.append({**cfg, "val_acc": best_val, "test_acc": test_acc})
            histories.append(h)
            models.append(m)

            self.log_write(
                f"[{i + 1}] hidden={cfg['hidden']:>3}  act={cfg['activation']:<6}  "
                f"opt={cfg['optimizer']:<8}  bs={cfg['batch_size']:>3}  ep={cfg['epochs']:>4}  |  "
                f"val_acc={best_val:.4f}  test_acc={test_acc:.4f}"
            )

        # лучшая по val_accuracy
        best_idx = int(np.argmax([r["val_acc"] for r in results]))
        best = results[best_idx]
        best_model = models[best_idx]

        self.log_write("")
        self.log_write(
            f"Лучшая модель: #{best_idx + 1}  "
            f"(hidden={best['hidden']}, act={best['activation']}, "
            f"opt={best['optimizer']}, bs={best['batch_size']}, ep={best['epochs']})"
        )
        self.log_write(
            f"val_acc={best['val_acc']:.4f}   test_acc={best['test_acc']:.4f}"
        )

        # метрики лучшей модели
        y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
        y_true = np.argmax(y_test, axis=1)
        self.log_write("\nConfusion matrix:")
        self.log_write(str(confusion_matrix(y_true, y_pred)))
        self.log_write("\nClassification report:")
        self.log_write(classification_report(y_true, y_pred, target_names=self.encoder.classes_))

        # ---------- графики ----------
        self.ax_acc.clear()
        self.ax_loss.clear()

        for i, h in enumerate(histories):
            is_best = (i == best_idx)
            color = "red" if is_best else "steelblue"
            lw = 2.5 if is_best else 1.0
            alpha = 1.0 if is_best else 0.35
            label = f"#{i + 1}" + (" (best)" if is_best else "")
            self.ax_acc.plot(h.history["val_accuracy"], color=color, lw=lw, alpha=alpha, label=label)
            self.ax_loss.plot(h.history["val_loss"],    color=color, lw=lw, alpha=alpha, label=label)

        self.ax_acc.set_title("Val accuracy")
        self.ax_acc.set_xlabel("epoch")
        self.ax_acc.set_ylabel("accuracy")
        self.ax_acc.grid(True, alpha=0.3)
        self.ax_acc.legend(fontsize=8)

        self.ax_loss.set_title("Val loss")
        self.ax_loss.set_xlabel("epoch")
        self.ax_loss.set_ylabel("loss")
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend(fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        self.status.config(text="Готово")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()