
import torch
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(__file__))

from resnet import ResNet6
from wrapper import init_all_attack_tables
from mcts import find_best_move

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet6().to(device)
model_path = os.path.join(os.path.dirname(__file__), '..', 'runs', 'resnet6_3M_games', 'bootstrap_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

init_all_attack_tables()

@app.route('/get_move', methods=['POST'])
def get_move():
    data = request.get_json()
    fen = data.get('fen')
    if not fen:
        return jsonify({'error': 'FEN string not provided'}), 400

    simulations = data.get('simulations', 2000)

    try:
        best_move = find_best_move(fen, simulations, model)
        return jsonify({'move': best_move})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8008, debug=True)
