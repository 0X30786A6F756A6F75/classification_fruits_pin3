import os
from flask import Flask, jsonify, request, url_for
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Passar o path do modelo a ser usado
model = joblib.load('randomForestClassifierModel.joblib')

@app.route('/classify', methods=['GET', 'POST'])
def classify_fruits():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'Erro': 'Nenhum arquivo foi enviado na request', 'status': 400})
    
        file = request.files['file']
        if file.filename == '':
            return jsonify({'Erro': 'Nenhum arquivo foi enviado na request'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
            # save_location = os.path.join('savedCsv', new_filename)
            # file.save(save_location)
        try:
            df = pd.read_csv(file)

            #Falta colocar as features
            features = df[['area','perimetro','eixo_maior','eixo_menor','excentricidade','eqdiasq','solidez','area_convexa','extensao','proporcao','redondidade','compactidade','fator_forma_1','fator_forma_2','fator_forma_3','fator_forma_4','RR_media','RG_media','RB_media','RR_dev','RG_dev','RB_dev','RR_inclinacao','RG_inclinacao','RB_inclinacao','RR_curtose','RG_curtose','RB_curtose','RR_entropia','RG_entropia','RB_entropia','RR_all','RG_all','RB_all']]

            predictions = model.predict(features)

            scores = {}
            for i in range(len(predictions)):

                if (predictions[i] in scores) :
                    scores[predictions[i]] += 1
                else :
                    scores[predictions[i]] = 1

            response = {
                'predictions': predictions.tolist(),
                'scores': scores
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'Erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
