from flask import Flask
from flasgger import Swagger
from flask_cors import CORS
from config.swagger import template, swagger_config
from routes.prediction_route import prediction

app = Flask(__name__)
port = 5000
CORS(app)


#Adicionar o bloco "definitions" no Swagger
app.config['SWAGGER'] = {
    'title': 'Predição de cura - API',
    'uiversion': 3
            }

swagger = Swagger(app, template=template, config=swagger_config)

# Registrando endpoints
app.register_blueprint(prediction)


# Executar o aplicativo Flask
if __name__ == '__main__':
    with app.app_context():
        app.run(host="0.0.0.0", port=port)

