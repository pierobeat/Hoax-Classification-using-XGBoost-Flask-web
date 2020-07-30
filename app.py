from jcopml.utils import load_model
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, template_folder="templates")
model = load_model("model/skripsi_xgb_pierre.pkl")

@app.route("/", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data = request.form['text']
        print("*"*100)
        print(data)
        pred = model.predict([data])
        print(pred)
        pred_proba = model.predict_proba([data])
        print(pred_proba)
        return render_template('index.html', sentiment=pred[0], sentiment_proba=pred_proba[0])
    return render_template('index.html', sentiment='', sentiment_proba='')

if __name__ == '__main__':
    app.run(port=5000, debug=True)




# @app.route("/", methods=['GET','POST'])
# def predict():
#     if request.method == 'POST':
#         data = request.form.get('text')
#         # Make prediction
#         df = pd.DataFrame([str(data)], columns=['content'])
#         print(df.head())
#         pred = model.predict(df)
#         print(pred)
#         return render_template('index.html', sentiment=pred['Klasifikasi_berita_hoax'][0])
#     return render_template('index.html', sentiment='')


         
        # {% if sentiment %}
        #     <p>The sentiment of the text is: {{sentiment}}</p>
        #     <p>The probability: {{sentiment_proba}}</p>
        # {% endif %}
