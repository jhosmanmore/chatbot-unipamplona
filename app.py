from flask import Flask, make_response, render_template, request, jsonify
from ChatBot import ChatBot  # Se importa la clase ChatBot

app = Flask(__name__)
bot = ChatBot()  # Se instancia la clase ChatBot

@app.route('/')
def index():
    #return render_template('index.html')
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    response = bot.ask(user_input)  # Usar el ChatBot para obtener la respuesta
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

