import requests

city_name='bhilai'
url = 'http://api.weatherapi.com/v1/current.json'
params = {'key': 'c7cfcc69ab69438a809181649232203', 'q': city_name.upper()}

response = requests.get(url, params=params)
if 'error' not in response.json():

    data = response.json()
    Result = dict()
    Result['humidity'] = data['current']['humidity']
    Result['wind_dir'] = data['current']['wind_dir']
    Result['wind_speed'] = data['current']['wind_mph']
    Result['tempreture'] = data['current']['temp_c']
    print(Result)
