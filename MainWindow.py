# This Python file uses the following encoding: utf-8
import sys

import numpy as np
from PySide2.QtCore import QThreadPool
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import NeuralNetwork
from WorkerThread import WorkerThread
from main_window import Ui_MainWindow


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        # Setup UI elements
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Click event connection
        self.ui.open_file_btn.clicked.connect(self.open_file_btn_clicked)
        self.ui.start_btn.clicked.connect(self.start_btn_clicked)
        self.ui.analyze_btn.clicked.connect(self.analyze_btn_clicked)

        # Class members
        self.is_file_selected = False
        self.file_path = None
        self.thread_pool = QThreadPool()
        self.batch_size = 32
        self.epoches = 500

        self.neuralNetwork = NeuralNetwork.NeuralNetwork(batch_size=self.batch_size, epoches=self.epoches)
        self.ui.learning_progress_bar.setMaximum(self.epoches-1)

    def analyze_btn_clicked(self):
        client_id = self.ui.client_id.text().strip()
        isDigit = client_id.isdigit()

        if client_id and isDigit:
            dateset = pd.read_csv('data_for_demonstration.csv')
            data = dateset.iloc[:, :].values
            is_valid_client = False
            for row in data:
                row_id = row[1]
                if row_id == int(client_id):
                    client = np.array2string(row, separator=" ")
                    self.show_client_data(row)
                    result_from_prediction = self.neuralNetwork.predict(row)
                    self.show_result_message_box(result_from_prediction)
                    is_valid_client = True
                    break

            if is_valid_client is False:
                msgBox = QMessageBox()
                msgBox.setWindowTitle("Грешка")
                msgBox.setText("Не е открит клиент")
                ret = msgBox.exec()
        else:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Грешка")
            msgBox.setText("Моля въведете валиден номер")
            ret = msgBox.exec()

    def open_file_btn_clicked(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, self.tr("Load Image"), self.tr("~/Desktop/"),
                                                       self.tr("CSV file(*.csv)"))
        if self.file_path is not None:
            self.is_file_selected = True
            self.ui.selected_file.setText(self.file_path)

    def start_btn_clicked(self):
        if self.is_file_selected is True:

            self.ui.open_file_btn.setEnabled(False)
            self.ui.start_btn.setEnabled(False)

            # Pass the function to execute
            worker = WorkerThread(self.file_path,
                                      self.neuralNetwork.learn)
            worker.signals.finished.connect(self.learning_complete)
            worker.signals.error.connect(self.learning_error)
            worker.signals.result.connect(self.learning_result)
            worker.signals.progress.connect(self.learning_progress_update)

            # Execute
            self.thread_pool.start(worker)

        else:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Грешка")
            msgBox.setText("Моля изберете файл")
            ret = msgBox.exec()

    def learning_result(self, history, confusion_matrix):
        # Plot the learning results
        plt.figure(1)
        plt.title("asdasd")
        plt.subplot(2, 1, 1)
        plt.plot(history.history['acc'])
        plt.title('Резултати от обучение')
        plt.ylabel('точност')

        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.xlabel('epoch')
        plt.ylabel('грешка')

        self.plot_confusion_matrix(confusion_matrix)
        plt.show()

    def learning_progress_update(self, epoch):
        self.ui.learning_progress_bar.setValue(epoch)

    def learning_complete(self):
        self.ui.open_file_btn.setEnabled(True)
        self.ui.start_btn.setEnabled(True)
        print("Learn process complete !!!")

    def learning_error(self, exctype, value):
        msgBox = QMessageBox()
        msgBox.setWindowTitle("Грешка")
        msgBox.setText("Грешка по време на обучение!")
        ret = msgBox.exec()

    def show_result_message_box(self, result):
        res_string = '{0:0.2f}'.format(result * 100)
        msgBox = QMessageBox()
        msgBox.setWindowTitle("Резултат")
        msgBox.setText(res_string + "% вероятност да напусне")
        ret = msgBox.exec()

    def show_client_data(self, row):
        self.ui.customer_id.setText("Customer id: " + str(row[1]))
        self.ui.surname_lbl.setText("Surname: " + str(row[2]))
        self.ui.credit_score_lbl.setText("Credit score: " + str(row[3]))
        self.ui.geography_lbl.setText("Geography: " + str(row[4]))
        self.ui.gender_lbl.setText("Gender: " + str(row[5]))
        self.ui.age_lbl.setText("Age: " + str(row[6]))
        self.ui.tenure_lbl.setText("Tenure: " + str(row[7]))
        self.ui.balance_lbl.setText("Balance: " + str(row[8]))
        self.ui.num_of_products_lbl.setText("Number of products: " + str(row[9]))
        self.ui.has_cr_card_lbl.setText("Has credit card: " + str(row[10]))
        self.ui.is_active_member_lbl.setText("Is active member: " + str(row[11]))
        self.ui.estimated_salary_lbl.setText("Estimated salary: " + str(row[12]))
        self.ui.exited_lbl.setText("Exited: " + str(row[13]))

    def plot_confusion_matrix(self, cm,
                              cmap=None,
                              normalize=False):
        accuracy = np.trace(cm) / float(np.sum(cm))
        title="Тестови резултати"
        target_names = ['Остават', 'Напускат']
        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Реални стойности')
        plt.xlabel('Предвидени стойности \n Постигната точност: {:0.2f}%'.format(accuracy*100))


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
