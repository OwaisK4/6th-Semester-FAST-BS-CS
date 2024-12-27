# client.py
import socket

def run_client():
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    server_address = ('localhost', 12345)
    client_socket.connect(server_address)

    try:
        # Send a message to the server
        message = "Hello, Server!"
        client_socket.sendall(message.encode('utf-8'))

        # Wait for a response from the server
        response = client_socket.recv(1024)
        print(f"Received response: {response.decode('utf-8')}")
    finally:
        # Close the connection
        client_socket.close()

if __name__ == "__main__":
    run_client()
