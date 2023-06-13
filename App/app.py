import requests
from flask import Flask,render_template,request,Markup
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

from PIL import Image

from App.utils.disease import disease_dic
from App.utils.fertilizer import fertilizer_dic

app = Flask(__name__)

crop_recommendation_model_path='Models/RF.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))
model = keras.models.load_model('Models/plantDiseaseDetectionModel.h5')

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']


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


def predict_disease(image):
    resized_image = image.resize((256, 256))
    image_array = np.array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction[0])
    return disease_classes[predicted_class]


@ app.route('/')
def home():
    title = 'Agriify'
    return render_template('index.html', title=title)

@ app.route('/crop-jyotashi')
def cropjyotashi():
    title = 'Crop Jyotashi'
    return render_template('CropRecommendation.html', title=title)

@ app.route('/fertilizer')
def fertilizerrecommendation():
    title = 'Agriify - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

@app.route('/disease-predict')
def plantdoctor():
    title = 'Agriify - Fertilizer Suggestion'
    return render_template('disease.html', title=title)

@app.route('/live-analytics')
def liveanaytics():
    title = 'Agriify - Live Analytics Of Moisture'
    return render_template('Graph.html', title=title)

@ app.route('/fertilizer-predict', methods=['POST'])
def fertrecommend():
    title = 'Agriify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('utils/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

@ app.route('/crop-result', methods=['POST'])
def crop_prediction():
    title = 'Agriify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if getWeatherData(city) != None:
            data  = getWeatherData(city)
            temperature, humidity = data['tempreture'],data['humidity']
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('Crop.html', prediction=final_prediction,addidata=[temperature, humidity], title=title)

        else:

            return render_template('TryAgain.html', title=title)

@app.route('/disease-predict', methods=['POST'])
def diseaseprediction():
        title = 'Agriify - Disease Detection'

        if request.method == 'POST':
            # if 'file' not in request.files:
            #     # return redirect(request.url)
            file = request.files.get('file')
            if not file:
                return render_template('disease.html', title=title)
            try:
                img = Image.open(file)
                prediction = predict_disease(img)
                prediction = Markup(str(disease_dic[prediction]))
                return render_template('disease-result.html', prediction=prediction, title=title)
            except:
                pass


if __name__ == '__main__':
    app.run(debug=True)