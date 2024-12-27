# server.py
import socket

def run_server():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    server_address = ('localhost', 12345)
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(1)

    while True:
        # Accept a connection
        print("Waiting for a connection...")
        connection, client_address = server_socket.accept()

        try:
            # Receive the message from the client
            print(f"Connection from {client_address}")
            message = connection.recv(1024)
            print(f"Received message: {message.decode('utf-8')}")

            # Send a response back to the client
            response = "Message received."
            connection.sendall(response.encode('utf-8'))
        finally:
            # Close the connection
            connection.close()

if __name__ == "__main__":
    run_server()
