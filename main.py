from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd
import joblib
import pickle

nationality_vectorizer=joblib.load(open('countvect.pkl','rb'))
nationality_nv_model=joblib.load(open('naivebayess.pkl','rb'))

gender_vectorizer=joblib.load(open('gender_vectorizer.pkl','rb'))
gender_nv_model = joblib.load(open('naivebayes.pkl', 'rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gender')
def gender():
    return render_template('gender.html')

@app.route('/api')
def api():
    return render_template('api_docs.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

    if request.method == 'POST':
        namequery=request.form['namequery']
        data=[namequery]
    vect=nationality_vectorizer.transform(data).toarray()
    result=nationality_nv_model.predict(vect)

    return render_template('index.html',name=namequery.upper(),prediction=result)

@app.route('/predict_gender',methods=['GET','POST'])
def predict_gender():

    gender_vectorizer=joblib.load(open('gender_vectorizer.pkl','rb'))

    gender_nv_model = joblib.load(open('naivebayes.pkl', 'rb'))

    if request.method == 'POST':
        namequery=request.form['namequery']
        data=[namequery]
    vector=gender_vectorizer.transform(data).toarray()
    results=gender_nv_model.predict(vector)
    return render_template('gender.html',name=namequery.title(),prediction=results)

@app.route('/api/nationality/<string:name>')
def api_nationality(name):

    data=[name]

    vecto = nationality_vectorizer.transform(data).toarray()
    resulto = nationality_nv_model.predict(vecto)

    return jsonify({"orignal name: ":name,"Predictions : ":resulto[0]})

@app.route('/api/gender/<string:name>')
def api_gender(name):

    data=[name]

    vecter=gender_vectorizer.transform(data).toarray()
    resultos=gender_nv_model.predict(vecter)
    if resultos == ['0']:
        resultos='Female'
    elif resultos == ['1']:
        resultos='Male'

    return jsonify({"orignal name: ": name, "Predictions : ": resultos})

if __name__=='__main__':
    app.run(debug=True)