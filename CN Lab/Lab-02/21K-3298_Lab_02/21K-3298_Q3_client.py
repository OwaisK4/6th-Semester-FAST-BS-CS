import socket

while True:
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   s.connect(('localhost', 8000))
   # print(f"Connected to {s.getpeername()}")
   sent_text = input("Enter marks: ")
   if sent_text == "-1":
      s.send("Client closed".encode())
      s.close()
      break
   s.send(sent_text.encode())
   recv_text = s.recv(2048).decode()
   print(recv_text)
   s.close()