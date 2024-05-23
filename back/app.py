from flask import Flask, jsonify, request, url_for
import joblib
import pandas as pd
import json
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
Expected_features = ['area','perimetro','eixo_maior','eixo_menor','excentricidade','eqdiasq','solidez','area_convexa',
                     'extensao','proporcao','redondidade','compactidade','fator_forma_1','fator_forma_2','fator_forma_3',
                     'fator_forma_4','RR_media','RG_media','RB_media','RR_dev','RG_dev','RB_dev','RR_inclinacao','RG_inclinacao',
                     'RB_inclinacao','RR_curtose','RG_curtose','RB_curtose','RR_entropia','RG_entropia','RB_entropia','RR_all','RG_all','RB_all']
# Passar o path do modelo a ser usado
model = joblib.load('randomForestClassifierModel.joblib')

@app.route('/classify', methods=['POST'])
def classify_fruits():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'Erro': 'Nenhum arquivo foi enviado na request', 'status': 400})
    
        file = request.files['file']
        if file.filename == '':
            return jsonify({'Erro': 'Nenhum arquivo foi enviado na request'}), 400
        if file and allowed_file(file.filename):
            try:
                df = pd.read_csv(file)

                missing_features = [feature for feature in Expected_features if feature not in df.columns]
                if missing_features:
                    return jsonify({'Erro': f'Colunas ausentes no arquivo CSV: {missing_features}'}), 400

                features = df[Expected_features]

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

                with open('jsonScores/lastPostResponse.json', 'w', encoding='utf-8') as f:
                    json.dump(response, f, ensure_ascii=False, indent= 4)

                return jsonify(response)

            except Exception as e:
                return jsonify({'Erro': str(e)}), 500

@app.route('/modelScores', methods=['GET'])
def returnSavedJsonScores():
    try:
        if os.path.exists('jsonScores/lastPostResponse.json'):
            with open('jsonScores/lastPostResponse.json', 'r', encoding='utf=-8') as f:
                scores = json.load(f)
            return jsonify(scores)
        else:
            return jsonify({'Erro': 'arquivo não foi encontrado.'}), 404
    except Exception as e:
        return jsonify({'Erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
