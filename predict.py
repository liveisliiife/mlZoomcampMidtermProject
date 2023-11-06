import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = "model_midterm.bin"

with open(model_file,"rb") as f_in:    
    dv_full,model = pickle.load(f_in)

app = Flask("heart_attack_prediction")


@app.route("/predict",methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv_full.transform([customer])
    y_pred = model.predict_proba(X)[0,1]

    heart_attack_prediction_ = y_pred >= 0.5

    result = {
        "heart attack probability":float(y_pred),
        "heart attack prediction":bool(heart_attack_prediction_)
        }

    return jsonify(result)



if __name__ == "__main__":
    #from gevent.pywsgi import WSGIServer
    #http_server = WSGIServer(('', 9696), app)
    #http_server.serve_forever()
    app.run(debug=True,host="0.0.0.0",port=9696)