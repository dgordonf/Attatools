from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory, current_app, send_file
from flask_gtts import gtts
import re
import pandas as pd
from pymysql import DataError
import nmslib
from sklearn.feature_extraction.text import TfidfVectorizer
import os
pd.options.mode.chained_assignment = None
import random

from rapidfuzz import process, fuzz, distance, string_metric
from fillpdf import fillpdfs
import urllib.request

from glob import glob
from io import BytesIO
from zipfile import ZipFile
import os
import datetime

app = Flask(__name__)

@app.route('/favicon.png') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.png', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods = ['GET'])
def landing():
    return render_template('landing.html')

@app.route('/supermatch', methods = ['POST', 'GET'])
def supermatch_app():
    return render_template('supermatch.html')

@app.route('/backend/supermatch', methods = ['POST','GET'])
def supermatch_back():
    
    if request.method == 'POST':

        data_1 = request.form.get('json_data_1')
        data_2 = request.form.get('json_data_2')

        #convert json to pandas dataframe
        df_1 = pd.read_json(data_1)
        df_2 = pd.read_json(data_2)

        #rename the first column to 'text'
        df_1.rename(columns={df_1.columns[0]: 'Name1'}, inplace=True)
        df_2.rename(columns={df_2.columns[0]: 'Name2'}, inplace=True)

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
        df = pd.merge(df, df_2, how = 'left', on = ['Name2'])
        # # #Convert to json
        df = df.to_json(orient='records')

    else: 
        df = '{}'

    return jsonify(df)

@app.route('/construction-forms', methods = ['GET'])
def construction_forms_app():
    return render_template('construction-forms.html')

@app.route('/backend/construction-forms-backend-1', methods = ['POST','GET'])
def construction_forms_back_1(): 
    #get all form data submited
    if request.method == 'POST':
        #get all form data on construction forms
        data = request.form.to_dict()

        #store names on "on" checkboxes into a list
        forms = []
        for key, value in data.items():
            if value == 'on':
                forms.append(key)

        df = pd.DataFrame()
        for form in forms:
            form_name = form + '.pdf'
            form_url = "static/construction-form-pdfs/" + form_name
            
            #Get all fillable forms
            json = fillpdfs.get_form_fields(form_url)

            #convert into a dataframe
            df_json = pd.DataFrame(json, index = [0])

            df_temp = pd.melt(df_json)
            df_temp['form_name'] = form

            #reorder columns
            df_temp = df_temp[['form_name', 'variable', 'value']]
            df_temp.columns = ['form_name', 'field_name', 'value']

            #add current index as a column
            df_temp['form_index'] = df_temp.index

            #append to dataframe
            df = df.append(df_temp)

        #Summarize the dataframe
        df_summary = df.groupby('field_name').agg({'form_name':'count','form_index':'sum'})

        df = df.groupby('field_name').form_name.apply(list)

        #left join
        df = pd.merge(df, df_summary, how = 'left', on = 'field_name')

        #Rename form_name_x to form_name
        df.rename(columns = {'form_name_x': 'form_name', 'form_name_y': 'form_name_count'}, inplace = True)

        #convert form_name_x to string
        df['form_name_string'] = df['form_name'].astype(str)

        ##Order by form_name_y decending, form_name_x acending, form_index decending
        df = df.sort_values(by = ['form_name_count', 'form_name_string', 'form_index'], ascending = [False, False, True])

        #drop form_name_string
        df = df.drop(columns = ['form_name_string'])

        #add field_name to dataframe
        df['field_name'] = df.index

        #Create JSON of data
        df_json = df.to_json(orient = 'records')

    else: 
        df_json = '{}'

    return df_json

@app.route('/backend/construction-forms-backend-2', methods = ['POST','GET'])
def construction_forms_back_2():
    #get all form data submited
    if request.method == 'POST':
        #get all form data on construction forms
        data = request.form.to_dict()

        #convert to a dataframe
        df = pd.DataFrame(data.items(), columns = ['field_name', 'value'])
        
        df['form_name'] = df['field_name'].apply(lambda x: x.split(' | ')[0])
        df['field_name'] = df['field_name'].apply(lambda x: x.split(' | ')[1])

        #Get unique forms
        forms = df.form_name.unique()
        forms = [x.split(',') for x in forms]
        forms = [item for sublist in forms for item in sublist]
        forms = list(set(forms))

        new_file_names = []
        folder = 'static/construction-form-pdfs/'
        filled_folder = 'static/filled_construction_form_pdfs/'
        for form in forms:
            #subset df on if form appears in form_name with a grepl
            df_form = df[df['form_name'].str.contains(form)]

            dict_form = dict(zip(df_form.field_name, df_form.value))

 
            name = form + '.pdf'
            name_filled = name.replace('.pdf', '_filled.pdf')

            #add new file name to list
            new_file_names.append(name_filled)

            #fill pdf
            fillpdfs.write_fillable_pdf(folder + name, filled_folder + name_filled, dict_form)

        stream = BytesIO()
        with ZipFile(stream, 'w') as zf:
            for file in new_file_names:
                zf.write(filled_folder + file, file)
        stream.seek(0)

        #zip name plus date
        zip_name = 'construction_forms_filled' + str(datetime.date.today()) + '.zip'

        return send_file(
            stream,
            as_attachment=True,
            download_name=zip_name
        )

    else:
        return '{}'


if __name__ == '__main__':
    #Need to make this port 443 in prod
    app.run(debug=True, use_reloader = True)