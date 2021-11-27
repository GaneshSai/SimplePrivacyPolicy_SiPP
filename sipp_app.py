import os
from flask import Flask, render_template, request, url_for, redirect
import json
from data.sipp_policies.sipp_policies import return_sipp

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

app = Flask(__name__)

@app.route('/')
def home(): 
    return render_template('home.html')

@app.route('/ViewSiPP', methods = ['POST', 'GET'])
def sipp():
    if request.method == 'POST': 
        policy_req = request.form['policy_req']
        sipp_pol_final = return_sipp(policy_req)
        op_json = json.dumps(sipp_pol_final)
        #op_json = json.dumps(sample_op)
        print('op_json'+str(type(op_json)))
        return render_template('sipp_op.html', op_json = op_json)
    return render_template("ViewSiPP.html")

@app.route('/guide')
def guide(): 
    return render_template('guide.html')

@app.route('/viewPolicy')
def view_policy(): 
    return render_template('viewPolicy.html')

if __name__ == "__main__": 
    app.run(host="localhost", port = 8000, debug = True, use_reloader=False)
