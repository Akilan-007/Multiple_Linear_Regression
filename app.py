from flask import Flask, render_template, request
import pickle
import numpy as np

sv = pickle.load(open('MLR.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    
    data1 = request.form['AT']
    data2 = request.form['V']
    data3 = request.form['AP']
    data4 = request.form['RH']
    tot_data = [[data1,data2,data3,data4]]
    arr = np.array(tot_data,dtype=float)
    pred = sv.predict(arr)
    PE = float(pred)
    html_content = f"<html><head></head><body style='background-color:green'><center><br><br><h1> MACHINE LEARNING PREDICTION</h1><br><br><h1> MULTIPLE LINEAR REGRESSION </h1><br><br><h1> The Predicted PE is {PE} </h1></center></body></html>"
    with open("templates\prediction.html",'w') as html_file:
        html_file.write(html_content)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
