# Filename: hello.py

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



from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar, QToolBar

class Window(QMainWindow):
    '''Main window'''
    def __init__(self, parent = None):
       
        
        def __createMenu(self):
            self.menu = self.menuBar().addMenu('&Menu')
            self.menu.addAction('&Exit', self.close)
            
        def __createToolBar(self):
            tools = QToolBar()
            self.addToolBar(tools)
            tools.addAction('Exit', self.close)
            
        def __createStatusBar(self):
            status = QStatusBar()
            status.showMessage('This is the status bar')
            self.setStatusBar(status)
        
        super().__init__(parent)
        self.setWindowTitle('QMainWindow')
        self.setCentralWidget(QLabel('This is the central widget'))
        __createMenu(self)
        __createToolBar(self)
        __createStatusBar(self)
            

if __name__ == '__main__':
    application = QApplication(sys.argv)
    main_window = Window()
    main_window.show()
    sys.exit(application.exec_())