# ============================================================
# MOTEUR IA - RECOMMANDATION SPORTIVE PERSONNALISÉE
# Modèle utilisé : meta-llama/Llama-3.2-3B-Instruct
# ============================================================


from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json
import torch

# CHARGEMENT DU MODÈLE

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("chargement du modèle IA…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


# CALCUL DE DURÉE (date début / date fin)

def calculer_nombre_semaines(date_debut: str, date_fin: str) -> int:
    d1 = datetime.fromisoformat(date_debut)
    d2 = datetime.fromisoformat(date_fin)

    diff_jours = (d2 - d1).days
    diff_semaines = max(1, diff_jours // 7)

    return diff_semaines


# RECOMMANDATION

def generer_recommandations(profil, objectif, restrictions, materiel, exercices):
    duree_semaines = calculer_nombre_semaines(
        objectif["date_debut"],
        objectif["date_fin"]
    )

    exercices_txt = "\n".join([
        f"- {ex['nom']} — muscle : {ex['muscle_principal']} — matériel : "
        f"{', '.join(ex.get('materiel', [])) if ex.get('materiel') else 'aucun'}"
        for ex in exercices
    ])

    # Prompt
    prompt = f"""
Tu es une IA experte en sport et en planification d'entraînement personnalisé.

### PROFIL UTILISATEUR
- Âge : {profil['age']}
- Taille : {profil['taille_cm']} cm
- Niveau d'activité : {profil['niveau_activite']}

### OBJECTIF PERSONNEL
- Type d'objectif : {objectif['type']}
- Valeur cible : {objectif['valeur_cible']} {objectif['unite']}
- Période : du {objectif['date_debut']} au {objectif['date_fin']}
- Durée totale : {duree_semaines} semaines

### RESTRICTIONS ALIMENTAIRES
{', '.join(restrictions) if restrictions else 'Aucune'}

### MATÉRIEL DISPONIBLE
{', '.join(materiel)}

### EXERCICES DISPONIBLES (depuis la base NoSQL)
{exercices_txt}

### MISSION
Génère un JSON STRICT contenant :
1. Un programme sportif hebdomadaire (3 niveaux : facile, normal, intensif)
2. La liste des exercices pertinents (déduits du matériel disponible)
3. Un plan de progression chiffré semaine par semaine ({duree_semaines} points)

IMPORTANT :
- Répond UNIQUEMENT en JSON valide.
- Pas de texte en dehors du JSON.
"""

    # Envoi au modèle
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_length=1800,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extraction
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        contenu_json = response[start:end]
        return json.loads(contenu_json)

    except Exception:
        return {
            "error": "Impossible de parser la réponse",
            "raw": response
        }

# TEST LOCAL

if __name__ == "__main__":
    profil = {
        "age": 28,
        "taille_cm": 178,
        "niveau_activite": "débutant"
    }

    objectif = {
        "type": "perte de poids",
        "valeur_cible": -5,
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

    resultat = generer_recommandations(profil, objectif, restrictions, materiel, exercices)

    print(json.dumps(resultat, indent=2, ensure_ascii=False))