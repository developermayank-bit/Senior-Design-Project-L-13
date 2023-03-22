import requests
from flask import Flask,render_template

app = Flask(__name__)


def getWeatherData(city_name):

    url = 'http://api.weatherapi.com/v1/current.json'
    params = {'key': 'c7cfcc69ab69438a809181649232203', 'q': city_name.upper()}

    response = requests.get(url, params=params)
    if 'error' not in response.json():
        data = response.json()
        Result = dict()
        Result['humidity']=data['current']['humidity']
        Result['wind_dir']=data['current']['wind_dir']
        Result['wind_speed']=data['current']['wind_mph']
        Result['tempreture']=data['current']['temp_c']
        return Result
    else:
        return None

@ app.route('/')
def home():
    title = 'Agrify'
    return render_template('index.html', title=title)

if __name__ == '__main__':
    app.run(debug=False)