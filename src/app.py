import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# 1. Gestion dynamique des chemins (indispensable sous Windows)
dossier_src = Path(__file__).resolve().parent  # C:\...\projet_python\src
dossier_racine = dossier_src.parent  # C:\...\projet_python

# Sécurité pour que Python trouve ton dossier 'src' et ses modules
if str(dossier_racine) not in sys.path:
    sys.path.insert(0, str(dossier_racine))

# On définit explicitement où Flask DOIT chercher les fichiers HTML
dossier_templates = dossier_racine / "templates"

# 2. Initialisation de Flask avec le chemin absolu forcé
app = Flask(__name__, template_folder=str(dossier_templates))

# Import de ton modèle de Machine Learning
try:
    from src.model import predict_single
except ImportError:
    # Fonction de secours si ton modèle n'est pas encore lié
    def predict_single(data_dict):
        return 35000.0


@app.route('/')
def home():
    # Flask va maintenant pointer directement dans le bon dossier racine/templates/
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        features = {
            "BHK": int(data.get("bhk", 2)),
            "Size": int(data.get("size", 800)),
            "floor_num": int(data.get("floor", 2)),
            "Bathroom": int(data.get("bath", 2)),
            "City": data.get("city"),
            "Area Type": data.get("area"),
            "Furnishing Status": data.get("furnish"),
            "Tenant Preferred": data.get("tenant"),
            "Point of Contact": data.get("contact")
        }

        predicted_rent = predict_single(features)
        min_rent = int(predicted_rent * 0.9)
        max_rent = int(predicted_rent * 1.1)

        return jsonify({
            "success": True,
            "rent": f"₹ {int(predicted_rent):,}".replace(",", " "),
            "range": f"Fourchette : ₹ {min_rent:,} - ₹ {max_rent:,}".replace(",", " ")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    print(f"[*] Dossier templates configuré sur : {dossier_templates}")
    app.run(debug=True, port=5000)