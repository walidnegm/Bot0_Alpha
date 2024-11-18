import socket
import json
import time

class SocketHelper:
    def __init__(self, host, port, max_retries=3, retry_delay=3):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client_socket = None



    def connect(self):

        retry_count = 0
        connected = False

        while not connected and retry_count < self.max_retries:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                connected = True
                print(f"Connected to {self.host}:{self.port}")
                return True

            except Exception as e:
                print(f"Connection attempt failed {e}")
                retry_count += 1
                if retry_count < self.max_retries:  
                    print("Retrying...")
                    time.sleep(self.retry_delay)
        print("Max retries reached")
        return False


    def send_data(self, data):
        if not self.client_socket:
            print("Error: Socket not connected.")
            if not self.connect():
                return False
        try:
            self.client_socket.sendall(bytes(json.dumps(data), 'utf-8'))
            print("Data sent:", data)
            return True

        except Exception as e:
            print(f"Error sending data: {e}")
            if not self.connect():
                return False
        
    def close_connection(self):
        if self.client_socket:
            self.client_socket.close()
            print("Connection closed.")

        else:
            print("Error: Socket not connected.")