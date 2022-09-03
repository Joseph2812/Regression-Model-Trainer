from multiprocessing.connection import _ConnectionBase
from enum import IntEnum, unique
import matplotlib.pyplot as plt

class PlotterProcess:
    def __call__(self, pipe:_ConnectionBase, title:str, xlabel:str, ylabel:str):
        self.__pipe = pipe
        self.__title = "[LIVE] " + title
        self.__xlabel = xlabel
        self.__ylabel = ylabel

        self.__x:list[float] = []
        self.__y:list[float] = []
        self.__figure, self.__axes = plt.subplots()      

        timer = self.__figure.canvas.new_timer(interval=1000)
        timer.add_callback(self.__callback)
        timer.start()

        self.__update_plot()
        self.__pipe.send(True) # Notify manager that the process has started

        plt.show()

    def __callback(self) -> bool:
        while self.__pipe.poll():
            command = self.__pipe.recv()

            if command[CommandType.FINISHED]:
                self.__terminate()
                return False
            else:
                if command[CommandType.X] == None or command[CommandType.Y] == None:
                    self.__x.clear()
                    self.__y.clear()
                else:
                    self.__x.append(command[CommandType.X])
                    self.__y.append(command[CommandType.Y])
                    self.__update_plot()
                
        self.__figure.canvas.draw()
        return True

    def __update_plot(self):
        self.__axes.clear()
        self.__axes.set_title(self.__title)
        self.__axes.set_xlabel(self.__xlabel)
        self.__axes.set_ylabel(self.__ylabel)
        self.__axes.plot(self.__x, self.__y, marker=".")

    def __terminate(self):
        plt.close(self.__figure)

@unique
class CommandType(IntEnum):
    FINISHED = 0
    X = 1
    Y = 2