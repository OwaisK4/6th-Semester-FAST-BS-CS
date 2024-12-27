# import socket
# import threading
# from cryptography.fernet import Fernet

# # Generate a key for Fernet encryption
# key = Fernet.generate_key()
# cipher_suite = Fernet(key)

# def handle_client(client_socket):
#     while True:
#         try:
#             # Receive message
#             msg = client_socket.recv(1024)
#             if msg == b'':
#                 break
#             # Decrypt message
#             decrypted_msg = cipher_suite.decrypt(msg)
#             print(f"Received: {decrypted_msg.decode('utf-8')}")
#             # Relay message to other clients
#             for client in clients:
#                 if client != client_socket:
#                     client.send(msg)
#         except:
#             client_socket.close()
#             clients.remove(client_socket)
#             break

# def run_server():
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.bind(('localhost', 12345))
#     server.listen(5)
#     print("Server is running...")
#     global clients
#     clients = []
#     while True:
#         client_socket, addr = server.accept()
#         print(f"Connected with {str(addr)}")
#         clients.append(client_socket)
#         thread = threading.Thread(target=handle_client, args=(client_socket,))
#         thread.start()

# if __name__ == "__main__":
#     run_server()

import socket
import threading

clients = {}

def on_new_client(clientsocket, addr):
    while True:
        msg = clientsocket.recv(1024)
        if msg:
            # Assuming the first message from a client is its identifier
            if addr not in clients:
                clients[addr] = msg.decode('utf-8')
            else:
                print(f"<{clients[addr]}> {msg.decode('utf-8')}")
                # Here you can add logic to send the message to a specific client
                # For example, to send a message to a specific client:
                # send_to_specific_client(msg, clients[addr])
        else:
            clientsocket.close()
            del clients[addr]
            break

def send_to_specific_client(message, client_identifier):
    for addr, identifier in clients.items():
        if identifier == client_identifier:
            clients[addr].send(message)
            break

s = socket.socket()
host = "localhost"
port = 50000

print('Server started!')
print('Waiting for clients...')

s.bind((host, port))
s.listen(5)

while True:
    c, addr = s.accept()
    threading.Thread(target=on_new_client, args=(c, addr)).start()
