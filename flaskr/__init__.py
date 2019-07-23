from flask import Flask, render_template, url_for, request
import numpy as np
import pandas
import pickle

app = Flask(__name__)

model = pickle.load(file=open('iris_model.pkl','rb'))

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/result',methods=['POST','GET'])
def result():
      #result = [1,2,3,4]
      if request.method == 'POST':
          result = request.form
          #type(result)->dict
          features = [float(j) for i,j in result.items()]
          features = [np.array(features)]
          y_pred = model.predict(features)[0]

          if y_pred == 0:
              y_pred = 'Iris-setosa'
          elif y_pred == 1:
              y_pred = 'Iris-versicolor'
          elif y_pred == 2:
              y_pred = 'Iris-Verginica'

      return render_template("re.html",result = str(y_pred))
if __name__ == '__main__':
    app.run(debug=True,)
