from flask import Flask, render_template, request

app = Flask(__name__)

# variable declaration
privacy_pols_json = {
	"shopping": ["amazon", "flipkart", "myntra", "paytm mall"], 
	"social_media": ["facebook", "instagram", "linkedin", "reddit", "tictoc", "twitter", "instagram"], 
	"messaging": ["whatsapp", "telegram", "snapchat", "imessage", "signal", "discord"]
}

'''
Policies to be further Collected : myntra, paytm mall, instagram, reddit, tictoc, twitter, instagr, snapchat, imessage, signal, discord
''' 

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/guide')
def guide(): 
    return render_template('guide.html')

@app.route('/viewPolicy', methods = ['POST'])
def view_policy(): 
    policy = request.form['policy']
    policy_text = open('./privary_policy_text/'+ policy +'_privacy_policy.txt', 'r').read()
    return render_template('viewPolicy.html', organization = policy, policy_text = policy_text)

@app.route('/SiPP', methods = ['POST'])
def sipp(): 
    policy = request.form['policy_text']
    return render_template('sipp.html', policy_text = policy_text)

if __name__ == "__main__": 
    app.run(debug = True)
