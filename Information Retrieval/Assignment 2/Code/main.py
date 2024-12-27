import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QShortcut,
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt
import os
import pickle
from indexer import VectorSpaceModel


filepath = "/home/owaisk4/Win_backup/FAST NU assignments/Information Retrieval/Assignment 2/ResearchPapers"
saved_index = os.path.join(filepath, "vector_space_index.pkl")

if __name__ == "__main__":

    model: VectorSpaceModel
    if os.path.exists(saved_index):
        with open(saved_index, "rb") as f:
            model = pickle.load(f)
        print("Loaded vector space model from file")
    else:
        files = os.listdir(filepath)
        files = [os.path.join(filepath, file) for file in files]    
        model = VectorSpaceModel(files)
        print("Created vector space model from scratch")
        with open(saved_index, "wb") as f:
            pickle.dump(model, f)
    
    # query = "machine learning"
    # result = model.process_query(query)
    # if len(result) == 0:
    #     print("NIL")
    # else:
    #     print(result)

    # Create the application
    app = QApplication(sys.argv)
    window = QWidget()

    # Input and output labels
    label = QLabel("Enter query:")
    input_text = QLineEdit()
    output_label = QLabel("Output:")

    # Create button and connect it to the function
    button = QPushButton("Submit", window)

    # Function to get query result on each button press
    def get_answer(query) -> str:
        result = model.process_query(query)
        if len(result) == 0:
            return "NIL"
        else:
            return ", ".join(result)

    enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), window)
    enter_shortcut.activated.connect(button.click)
    button.clicked.connect(
        lambda: output_label.setText(f"Output: {get_answer(input_text.text())}")
    )

    # Set up the layout
    layout = QVBoxLayout()
    layout.addWidget(label)
    layout.addWidget(input_text)
    layout.addWidget(button)
    layout.addWidget(output_label)

    # Set the layout for the main window
    window.setLayout(layout)

    # Set up the main window
    window.setWindowTitle("Vector Space Model (21K-3298)")
    window.show()

    # Run the application
    sys.exit(app.exec_())