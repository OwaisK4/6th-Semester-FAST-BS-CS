import socket
from math import ceil

def getGrade(marks: int):
    grades = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F']
    x = max(0, 90 - marks)
    dist = ceil(x / 4)
    dist = min(len(grades) - 1, dist)
    return grades[dist]


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 8000))
s.listen(5)
print("Waiting for connection")
while True:
    c, addr = s.accept()
    print(f"Got connection from addr: {addr}")
    marks = c.recv(1024).decode()
    marks = int(marks)
    grade = getGrade(marks)

    c.send(f"Grade is: {grade}".encode())
    c.close()