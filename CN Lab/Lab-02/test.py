import os, socket

# hostname = socket.gethostname()
hostname = "google.com"
ipaddress = socket.gethostbyname(hostname)

# print(hostname)
# print(ipaddress)

host = socket.gethostbyaddr("8.8.8.8")
print(host)

def check_ports(a, b):
    count = 0
    for port in range(a,  b+1):
        try:
            print(f"Port: {port} => service name: {socket.getservbyport(port)}")
            count += 1
        except:
            continue
    return count

if __name__ == "__main__":
    open_ports = check_ports(1, 1000)
    print(f"Total no. of open ports: {open_ports}")