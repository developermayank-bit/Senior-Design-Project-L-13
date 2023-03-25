import requests

url = 'http://api.weatherapi.com/v1/current.json'
params = {'key': 'c7cfcc69ab69438a809181649232203', 'q': 'bhilai'.upper()}

response = requests.get(url, params=params)

print(response.json())

if 'error' not in response.json():
    print(True)
else:
    print(False)
