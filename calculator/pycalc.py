import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QGridLayout, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt

from functools import partial

__version__ = '0.1.0'
__author__ = 'Jevon K Morris'


ERROR_MSG = 'ERROR'

def evaluate_expression(expression):
    try:
        result = str(eval(expression, {}, {}))
    except:
        result = ERROR_MSG
    return result


class PyCalcUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window Properties
        self.setWindowTitle('Python Calculator')
        self.setFixedSize(655, 755)
        # Widgets
        self.overallLayout = QVBoxLayout()
        self.__centralWidget = QWidget(self)
        self.setCentralWidget(self.__centralWidget)
        self.__centralWidget.setLayout(self.overallLayout)
        # Buttons and Display
        # Will literally be made in the order they are called.
        self.__createDisplay()
        self.__createButtons()
        
    def __createButtons(self):
        self.buttons = {}
        button_layout = QGridLayout()
        buttons = {'7': (0, 0),
                   '8': (0, 1),
                   '9': (0, 2),
                   '/': (0, 3),
                   'C': (0, 4),
                   '4': (1, 0),
                   '5': (1, 1),
                   '6': (1, 2),
                   '*': (1, 3),
                   '(': (1, 4),
                   '1': (2, 0),
                   '2': (2, 1),
                   '3': (2, 2),
                   '-': (2, 3),
                   ')': (2, 4),
                   '0': (3, 0),
                   '00': (3, 1),
                   '.': (3, 2),
                   '+': (3, 3),
                   '=': (3, 4),}
        
        for button_text, position in buttons.items():
            self.buttons[button_text] = QPushButton(button_text)
            self.buttons[button_text].setFixedSize(100, 100)
            button_layout.addWidget(self.buttons[button_text], position[0], position[1])
            self.overallLayout.addLayout(button_layout)
            
            
    def __createDisplay(self):
        self.display = QLineEdit()
        self.display.setFixedHeight(105)
        self.display.setAlignment(Qt.AlignRight)
        self.display.setReadOnly(True)
        self.overallLayout.addWidget(self.display)
        
    def clearDisplay(self):
        self.setDisplayText('')
    
    def displayText(self):
        return self.display.text()
    
    def setDisplayText(self, text):
        self.display.setText(text)
        self.display.setFocus()

class PyCalcController():
    def __init__(self, model, view):
        self.__evaluate = model
        self.__view = view
        self.__connectSignals()
    
    def __calculateResult(self):
        result = self.__evaluate(expression=self.__view.displayText())
        self.__view.setDisplayText(result)
        
    def __buildExpression(self, sub_expression):
        if self.__view.displayText() == ERROR_MSG:
            self.__view.clearDisplay()
        
        expression = self.__view.displayText() + sub_expression
        self.__view.setDisplayText(expression)
        
    def __connectSignals(self):
        for btnText, btn in self.__view.buttons.items():
            if btnText not in {'=', 'C'}:
                btn.clicked.connect(partial(self.__buildExpression, btnText))
        
        self.__view.buttons['='].clicked.connect(self.__calculateResult)
        self.__view.display.returnPressed.connect(self.__calculateResult)        
        self.__view.buttons['C'].clicked.connect(self.__view.clearDisplay)

        
def main():
    python_calculator = QApplication(sys.argv)
    view = PyCalcUI()
    view.show()
    model = evaluate_expression
    PyCalcController(model = model, view = view)
    sys.exit(python_calculator.exec_())
    
if __name__ == '__main__':
    main()