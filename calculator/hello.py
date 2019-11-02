# Filename: hello.py
# Note to self: Migrate this to a notebook later.

import sys

# 1. Import `Qapplication` and all required widgets

# from PyQt5.QtWidgets import QApplication, QLabel, QWidget
# from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QWidget
# from PyQt5.QtWidgets import QApplication, QGridLayout, QPushButton, QWidget
# from PyQt5.QtWidgets import QApplication, QFormLayout, QLineEdit, QWidget

# application = QApplication(sys.argv)
# window = QWidget()

# # window.setWindowTitle('PyQt5 App')
# window.setWindowTitle('QHBoxLayout')
# ''' There also exists QVBoxLayout to give a vertical look ''' 
# # window.setGeometry(100, 100, 280, 80)
# # window.move(60, 15)
# hlayout = QHBoxLayout()
# ''' Left -> Top, Right -> Bottom '''
# hlayout.addWidget(QPushButton('Left'))
# hlayout.addWidget(QPushButton('Center'))
# hlayout.addWidget(QPushButton('Right'))
# # hello_message = QLabel('<h1>Hello World!</h1>', parent = window)
# # hello_message.move(60, 15)
# window.setLayout(hlayout)

# window.setWindowTitle('QGridLayout')
# layout = QGridLayout()
# layout.addWidget(QPushButton('Button 0, 0'), 0, 0)
# layout.addWidget(QPushButton('Button 0, 1'), 0, 1)
# layout.addWidget(QPushButton('Button 0, 2'), 0, 2)
# layout.addWidget(QPushButton('Button 1, 0'), 1, 0)
# layout.addWidget(QPushButton('Button 1, 1'), 1, 1)
# layout.addWidget(QPushButton('Button 1, 2'), 1, 2)
# layout.addWidget(QPushButton('Button 2, 0'), 2, 0)
# layout.addWidget(QPushButton('Button 2, 1'), 2, 1)
# layout.addWidget(QPushButton('Button 2, 2'), 2, 2)
# layout.addWidget(QPushButton('Button 3, 1 + 3 columns span'), 3, 0, 3, 3)
# window.setLayout(layout)

# window.setWindowTitle('QFormLayout')
# layout = QFormLayout()
# layout.addRow('Name: ', QLineEdit())
# layout.addRow('Age: ', QLineEdit())
# layout.addRow('Class: ', QLineEdit())
# layout.addRow('Likes: ', QLineEdit())
# layout.addRow('Dislikes: ', QLineEdit())
# window.setLayout(layout)
# '''QLineEdit() can possibly be stored and used later'''


# window.show()
# sys.exit(application.exec_())



# from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QVBoxLayout

# class Dialog(QDialog):
#     '''Dialog'''
#     def __init__(self, parent = None):
#         super().__init__(parent)
#         self.setWindowTitle('QDialog')
#         dialogue_layout = QVBoxLayout()
#         form_layout = QFormLayout()
#         form_layout.addRow('Name:', QLineEdit())
#         form_layout.addRow('Age:', QLineEdit())
#         form_layout.addRow('Class:', QLineEdit())
#         form_layout.addRow('Likes:', QLineEdit())
#         form_layout.addRow('Dislikes:', QLineEdit())
#         dialogue_layout.addLayout(form_layout)
#         buttons = QDialogButtonBox()
#         buttons.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
#         dialogue_layout.addWidget(buttons)
#         self.setLayout(dialogue_layout)

# if __name__ == '__main__':
#     application = QApplication(sys.argv)
#     dialogue = Dialog()
#     dialogue.show()
#     sys.exit(application.exec_())



# from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar, QToolBar

# class Window(QMainWindow):
#     '''Main window'''
#     def __init__(self, parent = None):
       
        
#         def __createMenu(self):
#             self.menu = self.menuBar().addMenu('&Menu')
#             self.menu.addAction('&Exit', self.close)
            
#         def __createToolBar(self):
#             tools = QToolBar()
#             self.addToolBar(tools)
#             tools.addAction('Exit', self.close)
            
#         def __createStatusBar(self):
#             status = QStatusBar()
#             status.showMessage('This is the status bar')
#             self.setStatusBar(status)
        
#         '''Need to call this stuff after the functions...?'''
#         super().__init__(parent)
#         self.setWindowTitle('QMainWindow')
#         self.setCentralWidget(QLabel('This is the central widget'))
#         __createMenu(self)
#         __createToolBar(self)
#         __createStatusBar(self)

# if __name__ == '__main__':
#     application = QApplication(sys.argv)
#     main_window = Window()
#     main_window.show()
#     sys.exit(application.exec_())



# from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

# # widget.signal.connect(slot_function)

# def greeting():
#     '''slot'''
#     if msg.text():
#         msg.setText('')
#     else:
#         msg.setText('Hello World')
# '''Note: msg.setText, i.e. QLabel, supports string formatting '''
        
# application = QApplication(sys.argv)
# window = QWidget()
# window.setWindowTitle('Signals and Slots')
# layout = QVBoxLayout()

# button = QPushButton('Greet')
# button.clicked.connect(greeting)
# layout.addWidget(button)
# msg = QLabel('')
# layout.addWidget(msg)
# window.setLayout(layout)
# window.show()
# sys.exit(application.exec_())

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QGridLayout, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtCore import *
from PyQt5.QtGui import *

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