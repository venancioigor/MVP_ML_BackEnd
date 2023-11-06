from flask import Blueprint, jsonify, request
from flasgger import swag_from
import traceback
import pickle

prediction = Blueprint("prediction", __name__, url_prefix="/api/machineLearning")

# Carregando o modelo
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


@prediction.route('/fazPredicao', methods=['POST'])
@swag_from('../docs/prediction/fazPredicao.yaml')
def verificar_resultado_solucao():
    try:
        data = request.json
        # Captura os dados da requisição e atribuindo a variável em questão
        cabelo_de_troll = data['Cabelo_de_Troll']
        chifre_de_minotauro = data['Chifre_de_Minotauro']
        chifre_de_unicornio = data['Chifre_de_Unicornio']
        dedos_de_duende = data['Dedos_de_Duende']
        escama_de_basilisco = data['Escama_de_Basilisco']
        garra_de_grifo = data['Garra_de_Grifo']
        lagrimas_de_sereia = data['Lagrimas_de_Sereia']
        pena_de_fenix = data['Pena_de_Fenix']
        po_de_fada = data['Po_de_Fada']
        pocao_da_bruxa = data['Pocao_da_Bruxa']
        presa_de_quimera = data['Presa_de_Quimera']
        sangue_de_dragao = data['Sangue_de_Dragao']
        tinta_de_kraken = data['Tinta_de_Kraken']

        predict_request  =   [
                            cabelo_de_troll, chifre_de_minotauro, chifre_de_unicornio,dedos_de_duende, 
                            escama_de_basilisco, garra_de_grifo, lagrimas_de_sereia, pena_de_fenix,
                            po_de_fada, pocao_da_bruxa, presa_de_quimera, sangue_de_dragao, tinta_de_kraken
                            ]
        
        # Asseguro que todos os dados sejam convertidos para float
        predict_request = [float(i) for i in predict_request]

        # Predição utilizando o modelo
        prediction = model.predict([predict_request])

        result = "cura" if prediction[0] == 1 else "morte"

        # Retorne a predição como resposta JSON
        return jsonify({'prediction': result})
        
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Erro ao fazer predição de cura.'}), 500






