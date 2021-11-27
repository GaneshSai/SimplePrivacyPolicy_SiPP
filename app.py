import os
from flask import Flask, render_template, request
import json
#from pyFiles.sipp_modeling import sipp_modeling, modeling

app = Flask(__name__)

sample_op = {
    "policy"  : {
        "Data Retention": ["Data Retention - A", "Data Retention - B"], 
        "First Party Collection/Use": ["First Party Collection/Use - A", "First Party Collection/Use - B"], 
        "Third Party Sharing/Collection": ["Third Party Sharing/Collection - A", "Third Party Sharing/Collection - B"], 
        "User Access, Edit and Deletion": ["User Access, Edit and Deletion - A", "User Access, Edit and Deletion - B"], 
        "User Choice/Control": ["User Choice/Control - A", "User Choice/Control - B"], 
        "Data Security": ["Data Security - A", "Data Security - B"]
    }
}

# Routes
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
    return render_template('viewPolicy.html')

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
    #op_json = sipp_modeling(policy_text)
    #op_json = json.dumps(op_json)
    op_json = json.dumps(sample_op)
    print(op_json)
    #return render_template('test.html', op_json = op_json)
    return render_template('sipp_op.html', op_json = op_json)

# Run application main
if __name__ == "__main__": 
    app.run(host="localhost", port = 8000, debug = True, use_reloader=False)
