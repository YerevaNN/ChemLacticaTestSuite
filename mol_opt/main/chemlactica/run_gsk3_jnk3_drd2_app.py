from flask import Flask, request, jsonify
from tdc import Oracle

app = Flask(__name__)

gsk3b_oracle = Oracle("gsk3b")
jnk3_oracle = Oracle("jnk3")
drd2_oracle = Oracle("drd2")


@app.route('/gsk3b', methods=['POST'])
def send_gsk3b():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, str):
        return jsonify({'error': 'Expected a string'}), 400  
    
    if 'gsk3b_oracle' not in globals().keys():
        global gsk3b_oracle
    
    return jsonify({'gsk3b_score': gsk3b_oracle(data)}), 200


@app.route('/jnk3', methods=['POST'])
def send_jnk3():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, str):
        return jsonify({'error': 'Expected a string'}), 400  
    
    if 'jnk3_oracle' not in globals().keys():
        global jnk3_oracle
    
    return jsonify({'jnk3_score': jnk3_oracle(data)}), 200


@app.route('/drd2', methods=['POST'])
def send_drd2():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, str):
        return jsonify({'error': 'Expected a string'}), 400  
    
    if 'drd2_oracle' not in globals().keys():
        global drd2_oracle
    
    return jsonify({'drd2_score': drd2_oracle(data)}), 200


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'hello'}), 200


# @app.route('/aaa', methods=['GET'])
# def hello():
#     return jsonify({'message': 'Hello, !'})

if __name__ == '__main__':
    app.run(debug=False, port=4300, host="0.0.0.0")