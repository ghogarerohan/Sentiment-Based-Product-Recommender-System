
# Flask file to connect the backend ML model with the frontend HTML code
#import flask
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
from model import Recommendation_engine

recommend = Recommendation_engine()
app = Flask(__name__)  

@app.route('/', methods = ['POST', 'GET'])
def home():
    flag = False 
    data = ""
    if request.method == 'POST':
        flag = True
        user = request.form["username"]
        data=recommend.Get_top_5_products(user)
    return render_template('index.html', data=data, flag=flag)


if __name__ == '__main__' :
    app.run(debug=True )  






