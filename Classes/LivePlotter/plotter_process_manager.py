import multiprocessing as mp
from Classes.LivePlotter.plotter_process import PlotterProcess

class PlotterProcessManager:
    def __init__(self, title:str, xlabel:str, ylabel:str):
        self.__manager_end_pipe, self.__process_end_pipe = mp.Pipe()

        mp.Process(
            target=PlotterProcess(),
            args=(self.__process_end_pipe, title, xlabel, ylabel),
            daemon=True
        ).start()

        # Wait for the process to fully startup
        started = False
        while not started:
            while self.__manager_end_pipe.poll():
                started = self.__manager_end_pipe.recv()

    def plot(self, x, y):
        if self.__process_end_pipe.closed: return
        
        self.__manager_end_pipe.send((False, x, y)) # (finished, x, y)
    
    def end_process(self):
        if self.__process_end_pipe.closed: return

        self.__manager_end_pipe.send((True, None, None)) # Terminate plot