# FLASK

# PREDICTION WITH WEB INTERFACE DISPLAY

from flask import Flask, request, redirect, render_template, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import json
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import joblib
import warnings
import mysql.connector
from pymongo import MongoClient
warnings.filterwarnings("ignore")

app = Flask(__name__)
df = pd.read_csv('train.csv')
app.config['upload_folder']='images'

# CONNECTING TO MYSQL ======================================================================================================================

sqldb = mysql.connector.connect(
    host = 'localhost',
    user = 'karinaps',
    passwd = 'Kimtaehyung1809',
    database = 'mobile_project'
)
mycursor = sqldb.cursor()

# CONNECTING TO MONGODB ====================================================================================================================

client_mongo = MongoClient('mongodb://localhost:27017/')
mobileproject_mongo = client_mongo['mobile_project']
mobile_coll = mobileproject_mongo['mobile_coll']

# HOME ROUTE ===============================================================================================================================

@app.route('/')
def home():
    return render_template('hometry.html')

# RESULT ROUTE =========================================================================================================================

@app.route('/post', methods = ['POST'])
def post():
    body = request.form
    ram = int(body['ram'])
    fc = int(body['fc'])
    pc = int(body['pc'])
    battery_power = int(body['battery_power'])
    scr = int(body['scr'])

    if scr == 1:
        px_width = 240
        px_height = 320
        scr = '240 x 320'
    elif scr == 2:
        px_width = 320
        px_height = 480
        scr = '320 x 480'
    elif scr == 3:
        px_width = 360
        px_height = 480
        scr = '360 x 480'
    elif scr == 4:
        px_width = 400
        px_height = 800
        scr = '400 x 800'
    elif scr == 5:
        px_width = 480
        px_height = 800
        scr = '480 x 800'
    elif scr == 6:
        px_width = 540
        px_height = 960
        scr = '540 x 960'
    elif scr == 7:
        px_width = 640
        px_height = 960
        scr = '640 x 960'
    elif scr == 8:
        px_width = 640
        px_height = 1136
        scr = '640 x 1136'
    elif scr == 9:
        px_width = 720
        px_height = 1280
        scr = '720 x 1280'
    elif scr == 10:
        px_width = 750
        px_height = 1134
        scr = '750 x 1134'
    elif scr == 11:
        px_width = 800
        px_height = 1280
        scr = '800 x 1280'
    elif scr == 12:
        px_width = 1080
        px_height = 1920
        scr = '1080 x 1920'
    elif scr == 13:
        px_width = 1080
        px_height = 2340
        scr = '1080 x 2340'
    elif scr == 14:
        px_width = 1440
        px_height = 2560
        scr = '1440 x 2560'
    elif scr == 15:
        px_width = 1440
        px_height = 3040
        scr = '1440 x 3040'

    prediksi = model.predict([[ram, fc, pc, battery_power, px_height,px_width]])
    prediksi = round(prediksi[0])

    if ram == 2000:
        ram = 2
    elif ram == 4000:
        ram = 4
    elif ram == 6000:
        ram = 6
    elif ram == 8000:
        ram = 8

    
    # Another Features Recommendation with Content Based Reccomendation ======================================================
    #=============== creating dataframe ===========================================
    df_rec = df[['int_memory', 'mobile_wt', 'price_range']]

    def mergeCol(i):
        return str(i['int_memory']) + ' ' + str(i['mobile_wt']) + str(i['price_range'])
    
    df_rec['features'] = df_rec.apply(mergeCol, axis=1)

    #=============== count vectorizer =================================================
    from sklearn.feature_extraction.text import CountVectorizer
    model_rec = CountVectorizer(tokenizer=lambda x: x.split(' '))
    matrixfeature = model_rec.fit_transform(df_rec['features'])
    features = model_rec.get_feature_names()
    # jmlfeatures = len(features)
    # eventfeatures = matrixfeature.toarray()

    # print(features)
    # print(jmlfeatures)
    # print(eventfeatures[1])

    #=============== cosinus similarity ================================================
    from sklearn.metrics.pairwise import cosine_similarity
    score = cosine_similarity(matrixfeature)

    rec = prediksi
    indexrec = df_rec[df_rec['price_range'] == rec].index.values[0]
    # print(indexrec)

    daftarscore = list(enumerate(score[indexrec]))
    # print(daftarscore)

    sortdaftarscore = sorted(
        daftarscore,
        key = lambda j: j[1],
        reverse = True
    )

    #=============== top 5 features reccomendation ================================================
    
    similarspecs = []
    for i in sortdaftarscore:
        if i[1] > 0:
            similarspecs.append(i)
    
    # print(similarspecs)

    import random
    rekomen = random.choices(similarspecs, k=3)
    # print(rekomen)

    listrekomen = []
    for i in rekomen:
        rekomen1 = {}
        j = 0
        while j < 5:
            rekomen1['int_memory'] = df_rec.iloc[i[0]]['int_memory'],
            rekomen1['mobile_wt'] = df_rec.iloc[i[0]]['mobile_wt']
            j += 1
        listrekomen.append(rekomen1)
    
    # print(listrekomen)
    # print(listrekomen[0]['mobile_wt'])
    # print(listrekomen['int_memory'])

    if prediksi == 0:
        prediksi1 = 'low cost'
    elif prediksi == 1:
        prediksi1 = 'medium cost'
    elif prediksi == 2:
        prediksi1 = 'high cost'
    elif prediksi == 3:
        prediksi1 = 'very high cost'

    # INPUT THE RESULT TO MYSQL DATABASE =====================================================================================================================================================================================================================================================================================================================================

    data = dict(body)
    data['cost_category'] = prediksi1
    data['internal_memory'] = rekomen1['int_memory'][0].item()
    data['mobile_weight'] = rekomen1['mobile_wt'].item()
    data['pixel_height'] = px_height
    data['pixel_width'] = px_width
    # hapus = mycursor.execute('delete from mobile_table')
    mulai = mycursor.execute('alter table mobile_table auto_increment = 1')
    query = "INSERT INTO `mobile_project`.`mobile_table` (`ram`, `front_camera`, `primary_camera`, `battery_power`,`pixel_height`, `pixel_width`, `cost_category`, `internal_memory`, `mobile_weight`) VALUES (%(ram)s, %(fc)s, %(pc)s, %(battery_power)s, %(pixel_height)s, %(pixel_width)s, %(cost_category)s, %(internal_memory)s, %(mobile_weight)s)"

    mycursor.execute(query, data)
    sqldb.commit()

    # print(data)

    # INPUT THE RESULT TO MONGODB DATABASE ======================================================================================================================================================================================================================================================================================================================================

    data1 = dict(body)
    data1['cost_category'] = prediksi1
    data1['internal_memory'] = rekomen1['int_memory'][0].item()
    data1['mobile_weight'] = rekomen1['mobile_wt'].item()
    data1['pixel_height'] = px_height
    data1['pixel_width'] = px_width
    del data1['scr']
    # print(data1)
    # hapus1 = mobile_coll.delete_many({})
    addNew = mobile_coll.insert(data1)

    #===============================================================================================================================================================================================================================================================================================================================================================================

    print('ram: ', ram, 'fc: ', fc,'pc: ', pc, 'battery_power: ', battery_power, 'scr: ', scr, 'prediksi: ', prediksi)

    return render_template('result.html', prediksi = prediksi1, ram = ram, fc = fc, pc = pc, battery_power = battery_power, scr = scr, df_rec = listrekomen)
    
# ERROR ROUTE =================================================================================================================================================================

@app.route('/error')
def errornotfound():
    return render_template('errortry.html')

#404 error handler============================================================================================================================================================
@app.errorhandler(404)
def notFound404(error):
    return render_template('errortry.html')

# ACTIVATE SERVER ====================================================================================================================================================================================

if __name__ == '__main__':
        model = joblib.load('model_ml')
        app.run(debug = True, port = 1234)









