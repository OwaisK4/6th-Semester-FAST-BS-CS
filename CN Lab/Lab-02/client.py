import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 8000))
print(s.recv(2048).decode())
s.close()