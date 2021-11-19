import os
from flask import Flask, render_template, request
from pyFiles.model import SiPP

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home(): 
    return render_template('home.html')

@app.route('/SiPP')
def sipp():
    return render_template("sipp.html")

@app.route('/guide')
def guide(): 
    return render_template('guide.html')

@app.route('/viewPolicy', methods = ['POST'])
def view_policy(): 
    # policy = request.form['policy']
    #policy_text = open('./privary_policy_text/'+ policy +'_privacy_policy.txt', 'r').read()
    return render_template('viewPolicy.html')#, policy_text = policy_text)

@app.route('/submit_policy', methods = ['POST'])
def submit(): 
    user_choice = request.form['user_choice']
    if user_choice == 'dd': 
        policy_text = request.form['policy_dd']
    elif user_choice == 'txt': 
        policy_text = request.form['policy_txt']
    elif user_choice == 'file': 
        if request.method == 'POST': 
            policy_file = request.files['policy_file']
            policy_filename = policy_file.filename
            policy_file.save(policy_filename)
            policy_text = open(policy_filename, 'r').read()
            os.remove(policy_filename)
    # CALL FUNCTION HERE
    op_json = SiPP(policy_text)
    print(type(op_json))
    return render_template('sipp_op.html', op_json = op_json)

if __name__ == "__main__": 
    app.run(host="localhost", port=8000, debug=True)
    #app.run(debug = True)