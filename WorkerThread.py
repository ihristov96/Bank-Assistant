import sys
import traceback
import keras
from PySide2.QtCore import QRunnable
from WorkerSignals import WorkerSignals


class WorkerThread(QRunnable, keras.callbacks.Callback):
    def __init__(self, file, fn):
        QRunnable.__init__(self)
        keras.callbacks.Callback.__init__(self)
        self.fn = fn
        self.file = file
        self.signals = WorkerSignals()

    def on_epoch_end(self, epoch, logs=None):
        self.signals.progress.emit(epoch)

    def run(self):
        try:
            history, confusion_matrix= self.fn(self.file, self)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(history, confusion_matrix)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
