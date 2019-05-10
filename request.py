import requests
import numpy as np
from config import Config

# Create point cloud
shape = (Config.batch_size, Config.points_number, 6)
point_clouds = np.zeros(shape, dtype=np.float32)
data = {'point_clouds': point_clouds.tolist()}

# Endpoint
pointnet_url = 'http://127.0.0.1:5000/api'
dgcnn_url = 'http://127.0.0.1:5001/api'
pointcnn_url = 'http://127.0.0.1:5002/api'

# Requests
response = requests.post(pointcnn_url, json=data)
point_features = np.array(response.json()['features'])
print(point_features.shape)

