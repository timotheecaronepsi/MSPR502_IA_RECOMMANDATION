# ============================================================
# MOTEUR IA - RECOMMANDATION SPORTIVE PERSONNALISÉE
# Modèle utilisé : microsoft/phi-3-mini-4k-instruct
# ============================================================


import requests
import json

HF_API_TOKEN = "TON_TOKEN_ICI"  # token HuggingFace 
MODEL = "microsoft/phi-3-mini-4k-instruct"

API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def generer_recommandations(profil, objectif, restrictions, materiel, exercices):

    semaines = 9

    exercices_txt = "\n".join([
        f"- {e['nom']} (muscle: {e['muscle_principal']}, matériel: {', '.join(e.get('materiel', [])) or 'aucun'})"
        for e in exercices
    ])

    prompt = f"""
Tu es une IA experte en sport et nutrition.
Retourne UNIQUEMENT un JSON strict et valide.

Structure attendue :
- programme : facile[], normal[], intensif[]
- exercices_retenus : liste
- progression :
    - pourcentage_objectif[{semaines}]
    - calories_hebdo_cibles[{semaines}]

Données :
âge={profil['age']}, taille={profil['taille_cm']}, niveau={profil['niveau_activite']}
objectif={objectif['type']} vers {objectif['poids_cible']}kg
exercices:
{exercices_txt}

Réponds UNIQUEMENT avec le JSON.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    data = response.json()

    if isinstance(data, list):
        generated = data[0].get("generated_text", "")
    else:
        generated = data.get("generated_text", "")

    try:
        start = generated.index("{")
        end = generated.rindex("}") + 1
        return json.loads(generated[start:end])
    except:
        return {"error": "JSON invalide", "raw": generated}


# ---------- TEST ----------
if __name__ == "__main__":
    profil = {"age": 28, "taille_cm": 178, "niveau_activite": "débutant"}

    objectif = {
        "type": "perte de poids",
        "poids_cible": 110,
        "unite": "kg",
        "date_debut": "2026-01-10T00:00:00",
        "date_fin": "2026-03-20T00:00:00"
    }

    restrictions = ["sans lactose"]
    materiel = ["tapis de course", "haltères"]
    exercices = [
        {"nom": "Squat", "muscle_principal": "jambes", "materiel": []},
        {"nom": "Développé couché", "muscle_principal": "poitrine", "materiel": ["haltères"]},
        {"nom": "Course", "muscle_principal": "cardio", "materiel": ["tapis de course"]}
    ]

    result = generer_recommandations(profil, objectif, restrictions, materiel, exercices)
    print(json.dumps(result, indent=2, ensure_ascii=False))