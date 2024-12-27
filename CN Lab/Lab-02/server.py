import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 8000))

s.listen(5)
print("Waiting for connection")
while True:
    c, addr = s.accept()
    print(f"Got connection from addr: {addr}")
    c.send("Thanks for connecting".encode())
    c.close()