import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("localhost", 9000))

s.listen(5)
print("Waiting for connection")
while True:
    c, addr = s.accept()
    print(f"Got connection from addr: {addr}")
    print(c.recv(1024).decode())
    text = input()
    c.send(text.encode())
    c.close()
