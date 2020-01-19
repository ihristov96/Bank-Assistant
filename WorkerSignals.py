import keras
import numpy
from PySide2.QtCore import QObject, Signal


class WorkerSignals(QObject):
    # Defines the signals available from a running worker thread.
    finished = Signal()
    result = Signal(keras.callbacks.History, numpy.ndarray)
    error = Signal(tuple)
    progress = Signal(int)
