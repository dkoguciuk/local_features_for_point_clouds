import os


class Config:

    batch_size = 32
    points_number = 1024
    classes_number = 40
    server_host = os.getenv('SOCKET_SERVER', 'http://localhost:3000')