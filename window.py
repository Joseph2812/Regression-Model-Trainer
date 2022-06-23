from tkinter import *
from tkinter import ttk
from gui_classes import *

class Window(Tk):
    def __init__(self):
        super().__init__()

        self.resizable(width=False, height=False)
        self.title("Regression Model Trainer Setup")

        DataPathFrame(self)

        self.mainloop()

Window()