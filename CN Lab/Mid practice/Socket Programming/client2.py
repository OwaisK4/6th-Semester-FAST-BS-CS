# import socket
# import select
# import sys

# if len(sys.argv) != 3:
#     print("Correct usage: script, IP address, port number")
#     exit()

# IP_address = str(sys.argv[1])
# Port = int(sys.argv[2])

# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.connect((IP_address, Port))

# while True:
#     sockets_list = [sys.stdin, server]
#     read_sockets, write_socket, error_socket = select.select(sockets_list, [], [])
#     for socks in read_sockets:
#         if socks == server:
#             message = socks.recv(2048)
#             print(message.decode('utf-8'))
#         else:
#             message = sys.stdin.readline()
#             server.send(message.encode('utf-8'))
#             sys.stdout.write("<You>")
#             sys.stdout.write(message)
#             sys.stdout.flush()
    
#     server.close()

import socket

ClientSocket = socket.socket()
host = '127.0.0.1'
port = 50000

print('Waiting for connection')
try:
    ClientSocket.connect((host, port))
except socket.error as e:
    print(str(e))

# Send the client's identifier as the first message
username = input("Enter your username: ")
ClientSocket.send(username.encode('utf-8'))

while True:
    Input = input('Say Something: ')
    ClientSocket.send(str.encode(Input))
    Response = ClientSocket.recv(1024)
    print(Response.decode('utf-8'))
