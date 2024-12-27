import socket

filename = r"D:\Owais\conversation_log.txt"

while True:
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   s.connect(('localhost', 8000))
   print(f"Connected to {s.getpeername()}")
   f = open(filename, "a")
   sent_text = input()
   if sent_text == "-1":
      s.send("Client closed".encode())
      s.close()
      f.close()
      break
   s.send(sent_text.encode())
   recv_text = s.recv(2048).decode()
   print(recv_text)
   f.write(f"Client: {sent_text}\n")
   f.write(f"Server: {recv_text}\n")
   f.write("\n")
   s.close()
   f.close()