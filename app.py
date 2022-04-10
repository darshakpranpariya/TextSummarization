import numpy as np
from flask import Flask, request, jsonify, render_template

# importing sys
import sys
# adding Folder_2 to the system path
sys.path.insert(0, '../code/')
import my_model as my_module
import imp
imp.reload(my_module)


app = Flask(__name__)

@app.route('/') 
def index(): 
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text = [x for x in request.form.values()]
    prediction = my_module.generate_text_summary(text)

    output = prediction

    return render_template('index.html', prediction_text='Summarize text is here:- \n {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
