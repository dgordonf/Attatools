from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory, current_app
from flask_gtts import gtts
import re
import pandas as pd
import nmslib
from sklearn.feature_extraction.text import TfidfVectorizer
import os
pd.options.mode.chained_assignment = None
import random

from rapidfuzz import process, fuzz, distance, string_metric

application = Flask(__name__)

@application.route('/favicon.png') 
def favicon(): 
    return send_from_directory(os.path.join(application.root_path, 'static'), 'favicon.png', mimetype='image/vnd.microsoft.icon')

@application.route('/', methods = ['POST', 'GET'])
def main():
    return render_template('page1.html')

@application.route('/supermatch', methods = ['POST','GET'])
def kindamatch():
    
    if request.method == 'POST':

        data_1 = request.form.get('json_data_1')
        data_2 = request.form.get('json_data_2')

        #convert json to pandas dataframe
        df_1 = pd.read_json(data_1)
        df_2 = pd.read_json(data_2)

        # #process distance on both        
        a = process.cdist(df_1['Name1'].str.lower(), df_2['Name2'].str.lower(), scorer=distance.JaroWinkler.similarity)
     
        a = pd.DataFrame(a, columns = df_2['Name2'])
        a['Name1'] = df_1['Name1']
        
        a = pd.melt(a, id_vars=['Name1'])
        
        # ##Group by name and get best match
        b = a.groupby('Name1', as_index=False)['value'].max()
        
        #Merge together
        c = pd.merge(b, a, how = 'left', on = ['Name1', 'value'])

        #convert value to a decimal of 2 decimal places
        c['value'] = c['value'].apply(lambda x: round(x, 2))

        # #Sort by MatchScore
        c = c.sort_values(by = ['value'], ascending = [False])
        
        # #Remove duplicates based on name column keep the first
        c = c.drop_duplicates(subset=['Name1'], keep='first')

        #Left join to get all data
        df = pd.merge(df_1, c, how = 'left', on = ['Name1'])

        # # #Convert to json
        df = df.to_json(orient='records')

    else: 
        df = '{}'

    return jsonify(df)

if __name__ == '__main__':
    #Need to make this port 443 in prod
    application.run(debug=True, use_reloader = True)