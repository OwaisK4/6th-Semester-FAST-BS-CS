import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind(('172.16.10.27', 8000))
s.bind(('localhost', 8000))

s.listen(5)
print("Waiting for connection")
while True:
    c, addr = s.accept()
    print(f"Got connection from addr: {addr}")
    print(c.recv(1024).decode())
    text = input()
    c.send(text.encode())
    c.close()