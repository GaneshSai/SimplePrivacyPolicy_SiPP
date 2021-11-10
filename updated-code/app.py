from io import TextIOWrapper
import torch
from transformers import EarlyStoppingCallback
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from flask import *
import nltk
from flask_session import Session
nltk.download('punkt')
app = Flask(__name__)
# from tansformers import
app.secret_key = "super secret key"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route('/')
def upload():
    return render_template("test.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        # print("enter")
        # if request.form.get("File"):
        Text="Choose \n any \n button to view data"
        if request.form.get("policy"):
            data = request.form.get("policy")
            # print(data)
            data = open('./privary_policy_text/'+ data +'_privacy_policy.txt', 'r').read()
            session['data'] = data
            show = [{
                'data': [data],
                'heading': "File Data"
            }]
            print(0,Text)
            return render_template("options.html", show=Text)
        else:
            f = request.files['policy']
            f.seek(0)
            data = (f.read().decode('utf-8'))
            session['data'] = data
            show = [{
                'data': [data],
                'heading': "File Data"
            }]
            print(1,Text)
            # print(session.get('data'))
            return render_template("options.html", show=Text)


@app.route('/manoj', methods=['POST'])
def caller():
    # print(session.get('data'))
    # print("enter")
    Categories = ['Data Retention', 'First Party Collection/Use',
                  'International and Specific Audiences', 'Other', 'Policy Change',
                  'Third Party Sharing/Collection', 'User Access, Edit and Deletion',
                  'User Choice/Control', 'Data Security', 'Do Not Track']
    Number = [0, 3, 4, 5, 6, 7, 8, 9, 1, 2]
    # Filter[10]
    Filter = []
    for i in range(10):
        Filter.append([])

    if request.method == 'POST':
        # data = request.args.get('', None)
        # print(data)
        show = []
        check = session.get('data')
        check = check.replace("\n", "")
        # print(check)

        toke_data = nltk.tokenize.sent_tokenize(check)

        # print(check)
        # print("sjds")
        # send=[]
        print("enter")

        for sent in toke_data:
            # for j in sent:
            try:
                print(sent)
                new = tokenizer(sent, padding=True, return_tensors='pt')
                # print(new)
                output = model(**new).logits
                # if output[0][0] < output[0][1]:
                arr = output[0].tolist()
                index1 = arr.index(max(arr))

                # print(index1)
                # print(output[0].tolist())
                # print(output[0][0][0],output[0][1])
                # print(sent)
                # show.append(sent)
                Filter[index1].append(sent)
            except:
                print("error occur")

                # pass

        if request.form.get("dataRetention"):
            send = [
                {
                    'data': Filter[0],
                    'heading':Categories[0]}
            ]
            return render_template("options.html", show=send)
        if request.form.get("security"):
            send = [
                {
                    'data': Filter[1],
                    'heading': Categories[8]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
        if request.form.get("collection"):
            send = [
                {
                    'data': Filter[3],
                    'heading': Categories[1]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
        if request.form.get("audiences"):
            send = [
                {
                    'data': Filter[4],
                    'heading': Categories[2]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
        if request.form.get("other"):
            send = [
                {
                    'data': Filter[5],
                    'heading': Categories[3]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
        if request.form.get("change"):
            send = [
                {
                    'data': Filter[6],
                    'heading': Categories[4]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
        if request.form.get("thirdparty"):
            send = [
                {
                    'data': Filter[7],
                    'heading': Categories[5]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
        if request.form.get("access"):
            send = [
                {
                    'data': Filter[8],
                    'heading': Categories[6]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
        if request.form.get("track"):
            send = [
                {
                    'data': Filter[9],
                    'heading': Categories[9]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
        if request.form.get("choice"):
            send = [
                {
                    'data': Filter[2],
                    'heading': Categories[7]
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
    # return render_template("check.html")


if __name__ == '__main__':
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        "checkpoint-4500", num_labels=10)
    app.run(debug=True)
