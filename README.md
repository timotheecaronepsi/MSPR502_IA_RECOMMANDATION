# IA Recommandation - Mode Programme Sportif (Distant)

Ce README documente premièrement le script `Ia_recom_mistral_distant.py`.

Le script genere un programme sportif personnalise en JSON, en appelant un modele distant via Hugging Face Router.

## Objectif du script

`Ia_recom_mistral_distant.py`:
- valide une requete utilisateur (niveau, objectif, periode, biometrie, materiels),
- calcule le nombre de semaines selon les dates,
- construit des objectifs intermediaires de poids,
- filtre les exercices compatibles avec le niveau et le materiel,
- appelle le modele LLM distant,
- retourne un JSON propre (ou une erreur explicite).

## Prerequis

- Python 3.10+
- Un token Hugging Face avec acces Router
- Les fichiers de donnees presents dans le projet:
	- `exercices.txt`
	- `materiels.txt`
	- `exercice_materiel.txt`

## Configuration `.env`

1. Copier `.env.example` vers `.env`
2. Renseigner les variables

Exemple:

```env
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
HF_TIMEOUT_SEC=45
HF_MAX_RETRIES=3
HF_MAX_TOKENS=600
```

Notes:
- `HF_API_TOKEN` est obligatoire.
- `HF_TIMEOUT_SEC` et `HF_MAX_RETRIES` influencent la robustesse.
- `HF_MAX_TOKENS` controle la taille max de reponse du modele.

## Lancer le script

Le script utilise le bloc `requete_test` en bas de fichier dans `if __name__ == "__main__":`.
Remplissez le et lancer le fichier python pour essayer.

## Contrat d'entree

Champs attendus:

- `niveau`: `facile` | `normal` | `intensif`
- `objectif`: `perte_de_poids` | `prise_de_masse`
- `date_debut`: format ISO 8601
- `date_fin`: format ISO 8601
- `valeur_cible`: nombre (kg)
- `unite`: doit etre `kg`
- `biometrie.poids_kg`: nombre
- `materiels`: liste de noms de materiels

Exemple:

```json
{
	"niveau": "intensif",
	"objectif": "prise_de_masse",
	"date_debut": "2026-01-10T00:00:00",
	"date_fin": "2026-03-20T00:00:00",
	"valeur_cible": 125,
	"unite": "kg",
	"biometrie": {"poids_kg": 115},
	"materiels": ["Banc d'entraînement", "Haltères", "Barres", "Rameur"]
}
```

## Sortie attendue

Le modele doit renvoyer un JSON de ce type:

```json
{
	"niveau": "intensif",
	"objectif": "prise_de_masse",
	"programme": [
		{
			"exercice": "Nom (Muscle)",
			"series": 4,
			"repetitions": 10,
			"temps_de_repos": 90
		}
	],
	"progression": {
		"nombre_semaines": 10
	}
}
```

En cas d'echec, le script retourne un objet avec `error`.

## Regles metier implementees

- Priorite aux exercices compatibles avec le materiel utilisateur.
- Fallback sur exercices sans materiel si insuffisant.
- Pour `perte_de_poids`, inclue plusieurs exercices cardio.

## Robustesse JSON et reseau

- Parsing robuste de la reponse modele (extraction du premier objet JSON valide).
- Retry de generation si JSON non trouve.
- Fallback de modeles (`HF_MODEL_FALLBACKS`) en cas d'indisponibilite.
- Metriques de run disponibles via `metrics_snapshot()`.

## Depannage rapide

- Erreur `HF_API_TOKEN manquant`:
	- verifier `.env`
- Erreur timeout:
	- augmenter `HF_TIMEOUT_SEC`
	- augmenter `HF_MAX_RETRIES`
- JSON invalide:
	- reduire la complexite du prompt
	- augmenter `HF_MAX_TOKENS`