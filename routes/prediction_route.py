from flask import Blueprint, jsonify, request
from flasgger import swag_from
import traceback
import pickle
import pandas as pd
import numpy as  np
from sklearn.preprocessing import StandardScaler

prediction = Blueprint("prediction", __name__, url_prefix="/api/machineLearning")

# Carregando o modelo
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)



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

        dados = {
                'Pena de Fenix': pena_de_fenix,
                'Chifre de Unicornio': chifre_de_unicornio,
                'Sangue de Dragão': sangue_de_dragao,
                'Lágrimas de Sereia': lagrimas_de_sereia,
                'Pó de Fada': po_de_fada,
                'Dedos de Duende': dedos_de_duende,
                'Poção da Bruxa': pocao_da_bruxa,
                'Garra de Grifo': garra_de_grifo,
                'Cabelo de Troll': cabelo_de_troll,
                'Tinta de Kraken': tinta_de_kraken,
                'Chifre de Minotauro': chifre_de_minotauro,
                'Escama de Basilisco': escama_de_basilisco,
                'Presa de Quimera': presa_de_quimera
                    }
        
        atributos = [
              'Pena de Fenix', 'Chifre de Unicornio', 'Sangue de Dragão','Lágrimas de Sereia',
              'Pó de Fada','Dedos de Duende','Poção da Bruxa', 'Poção da Bruxa', 'Garra de Grifo',
              'Cabelo de Troll',  'Tinta de Kraken','Chifre de Minotauro','Escama de Basilisco',
              'Presa de Quimera'
             ]
        entrada = pd.DataFrame([dados], columns=atributos)
        array_entrada = entrada.values
        X_entrada = array_entrada[:,0:13].astype(float)

        # Padronização nos dados de entrada usando o scaler utilizado em X
        rescaledEntradaX = scaler.transform(X_entrada)
        
        # Predição utilizando o modelo
        prediction = model.predict(rescaledEntradaX)

        result = "cura" if prediction[0] == 1 else "morte"

        # Retorne a predição como resposta JSON
        return jsonify({'prediction': result})
        
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Erro ao fazer predição de cura.'}), 500






