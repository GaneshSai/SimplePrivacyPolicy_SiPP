import os
from flask import Flask, render_template, request
import json
#from pyFiles.model import SiPP
from pyFiles.sipp_modeling import sipp_modeling, modeling

sample_op = {
    "policy"  : {
        "Data Retention": ["Data Retention - This is sample sentence AAA", "Data Retention - This is sample sentence BBB"],
        "First Party Collection/Use": ["First Party Collection/Use - This is sample sentence AAA", "First Party Collection/Use - This is sample sentence BBB"],
        "International and Specific Audiences": ["International and Specific Audiences - This is sample sentence AAA", "International and Specific Audiences - This is sample sentence BBB"],
        "Other": ["Other - This is sample sentence AAA", "Other - This is sample sentence BBB"],
        "Policy Change": ["Policy Change - This is sample sentence AAA", "Policy Change - This is sample sentence BBB"],
        "Third Party Sharing/Collection": ["Third Party Sharing/Collection - This is sample sentence AAA", "Third Party Sharing/Collection - This is sample sentence BBB"],
        "User Access, Edit and Deletion": ["User Access, Edit and Deletion - This is sample sentence AAA", "User Access, Edit and Deletion - This is sample sentence BBB"],
        "User Choice/Control": ["User Choice/Control - This is sample sentence AAA", "User Choice/Control - This is sample sentence BBB"],
        "Data Security": ["Data Security - This is sample sentence AAA", "Data Security - This is sample sentence BBB"],
        "Do Not Track": ["Do Not Track - This is sample sentence AAA", "Do Not Track - This is sample sentence BBB"]
    }
}

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home(): 
    print(type(sample_op))
    # modeling()
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
    op_json = sipp_modeling(policy_text)
    op_json = json.dumps(op_json)
    print(op_json)
    #return render_template('test.html', op_json = op_json)
    return render_template('sipp_op.html', op_json = op_json)

if __name__ == "__main__": 
    #os.system('./pyFiles/sipp_modeling.py')
    app.run(host="localhost", port = 8000, debug = True, use_reloader=False)
    #app.run(debug = True)