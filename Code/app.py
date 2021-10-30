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
    return render_template("index.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        # print("enter")
        # if request.form.get("File"):

        if request.form.get("file"):
            data = request.form.get("file")
            # print(data)
            session['data'] = data
            show = [{
                'data': [data],
                'heading': "File Data"
            }]
            return render_template("options.html", show=show)
        else:
            f = request.files['file']
            f.seek(0)
            data = (f.read().decode('utf-8'))
            session['data'] = data
            show = [{
                'data': [data],
                'heading': "File Data"
            }]
            # print(session.get('data'))
            return render_template("options.html", show=show)


@app.route('/manoj', methods=['POST'])
def caller():
    print(session.get('data'))
    print("enter")

    if request.method == 'POST':
        # data = request.args.get('', None)
        # print(data)
        show = []
        check = session.get('data')
        check = check.replace("\n", "")
        print(check)

        toke_data = nltk.tokenize.sent_tokenize(check)

        # print(check)
        # print("sjds")
        # send=[]

        if request.form.get("choice/consent"):
            for sent in toke_data:
                # for j in sent:
                try:
                    new = tokenizer(sent, padding=True, return_tensors='pt')
                    output = model(**new).logits
                    if output[0][0] < output[0][1]:
                        # print(sent)
                        show.append(sent)
                except:
                    pass
            send = [
                {
                    'data': show,
                    'heading': "choice/consent"}
            ]
            return render_template("options.html", show=send)
        if request.form.get("security"):
            send = [
                {
                    'data': ["Need model"],
                    'heading': "security"
                }
            ]
            # return "Rahul"
            return render_template("options.html", show=send)
    # return render_template("check.html")


if __name__ == '__main__':
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        "checkpoint-1500", num_labels=2)
    app.run(debug=True)
