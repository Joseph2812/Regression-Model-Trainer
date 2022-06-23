from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
import pandas as pd

class DataPathFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.file_path_var = StringVar(self, value="")
        self.data_selection_frame = None
        self.error_label = None

        self.grid(column=0, row=0)
              
        vcmd = (self.register(self.__validate_path), '%P')

        ttk.Label(self, text="Data Path:").grid(column=0, row=0, padx=10, pady=10)
        ttk.Entry(self, validate="focusout", validatecommand=vcmd, textvariable=self.file_path_var, width=100).grid(column=1, row=0, padx=(5, 10), pady=10)
        
        button = ttk.Button(self, text="Browse", command=self.__open_file_explorer)
        button.grid(column=2, row=0, padx=(0, 10), pady=10)
        master.bind("<Return>", lambda event: button.focus_set())

        ttk.Style().configure("Error.TLabel", foreground="red")
        self.error_label = ttk.Label(self, text="Invalid file path (must point to an existing .csv file)", style="Error.TLabel")
        self.error_label.grid(column=1, row=1, padx=10, pady=(0, 10))
        self.error_label.grid_remove()

    def __open_file_explorer(self):
        file_path = fd.askopenfilename(
            parent=self.master,
            title="Choose the training data",
            initialdir="./",
            filetypes=[("Comma Separated Values", "*.csv")]
        )
        self.file_path_var.set(file_path)
        self.__validate_path(file_path)
    
    def __validate_path(self, file_path) -> bool:
        try: 
            dataset = pd.read_csv(file_path)
        except:              
            self.error_label.grid()
            return False

        self.error_label.grid_remove()           

        if self.data_selection_frame == None:
            self.data_selection_frame = DataSelectionFrame(self.master, dataset)
        else:
            self.data_selection_frame.load_data(dataset)

        return True

class DataSelectionFrame(ttk.Frame):
    def __init__(self, master, dataset):
        super().__init__(master)

        self.active_buttons = [] #[{reference:ref, value:int}]

        self.grid(column=0, row=1)

        ttk.Label(self, text="Select which features to train on:").pack(side=TOP, anchor=N)
        self.load_data(dataset)
    
    def load_data(self, dataset):
        column_names = dataset.columns.values.tolist()

        for button in self.active_buttons:
            button["reference"].destroy()
            del button
        self.active_buttons.clear()

        for name in column_names:
            value = IntVar(self, 0)
            button = ttk.Checkbutton(self, text=f"{name}", offvalue=0, onvalue=1, variable=value)
            button.pack(side=LEFT, anchor=S)

            self.active_buttons.append({"reference": button, "value": value})