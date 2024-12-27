import socket


def check_port(port):
    try:
        service = socket.getservbyport(port)
        print(f"Port: {port} => service name: {service}")
    except:
        print(f"No service exists on given port {port}")


if __name__ == "__main__":
    while True:
        try:
            port = int(input("Enter port: "))
            check_port(port)
        except KeyboardInterrupt:
            break
