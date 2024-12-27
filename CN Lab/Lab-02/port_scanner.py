import os, socket
from time import time

startTime = time()

if __name__ == "__main__":
    # target = socket.gethostbyname('google.com')
    target = socket.gethostbyname("127.0.0.1")
    print(f"Starting scan on host: {target}")

    for i in range(50, 500):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = s.connect_ex((target, i))

        if conn == 0:
            print(f"Port: {i} OPEN")
        s.close()
    print(f"Time taken: {time() - startTime}")