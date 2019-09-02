from flask import Flask, render_template, url_for, request
import numpy as np
import pandas
import pickle
import os

# attaching images folder

app = Flask(__name__)

IMAGE = os.path.join('static','images')
app.config['UPLOAD_FOLDER'] = IMAGE

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
              c_pred = 'Iris-setosa'
          elif y_pred == 1:
              c_pred = 'Iris-versicolor'
          elif y_pred == 2:
              c_pred = 'Iris-Verginica'
      img = ['Iris_setosa.jpg','Iris_versicolor.jpg','Iris_virginica.jpg']
      image_file = os.path.join(app.config['UPLOAD_FOLDER'],img[y_pred])
      return render_template("re.html",result = str(c_pred), b_image=image_file)



if __name__ == '__main__':
    app.run(debug=True,)
