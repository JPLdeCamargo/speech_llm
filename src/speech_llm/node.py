from abc import ABC, abstractmethod

class Node(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def execute(self, input:any) -> any:
        pass
    