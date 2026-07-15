"""
Teste l'agent via l'API avec des prompts de complexité croissante, un tool
à la fois, pour isoler lequel bloque ou traîne. Affiche le temps écoulé
pour chaque appel — si un appel dépasse largement les autres, c'est lui
le coupable.
"""
import requests
import time

BASE_URL = "http://localhost:8000"
ENDPOINT_URL = f"{BASE_URL}/chat"   # <-- adaptez à votre route réelle

TESTS = [
    ("Météo seule (get_meteo_vigne)",
     "Quelle est la météo actuelle sur la parcelle irouleguy-pilote-01 ?"),

    ("Seuils seuls (get_seuils_alerte)",
     "Quels sont les seuils d'alerte phytosanitaires actuels ?"),

    ("Calcul seul (calcul_agronomique)",
     "Calcule la dose de traitement au cuivre pour une surface de 1.2 ha."),

    ("Data Commons seul (get_contexte_parcelle)",
     "Quel est l'historique de la parcelle irouleguy-pilote-01 ?"),

    #("RAG texte seul (rechercher_connaissance_phytosanitaire)",
     #"Que recommande la documentation pour traiter le mildiou ?"),

    #("Image Agent seul (diagnostiquer_image_maladie) — le plus susceptible d'être lent",
     #"Diagnostique cette image : data/Grapevine_img/train/Black Rot/img_5.jpg (parcelle irouleguy-pilote-01)"),
]


def tester_un_prompt(nom: str, prompt: str, timeout: int = 300):
    print(f"\n{'='*60}\n{nom}\n{'='*60}")
    print(f"Prompt: {prompt}")
    t0 = time.time()
    try:
        resp = requests.post(ENDPOINT_URL, json={"message": prompt, "thread_id": f"test-{nom[:20]}"}, timeout=timeout)
        duree = time.time() - t0
        print(f"⏱️  {duree:.1f}s — status {resp.status_code}")
        if resp.status_code == 200:
            print("Réponse:", resp.json())
        else:
            print("Erreur:", resp.text[:500])
    except requests.exceptions.ReadTimeout:
        duree = time.time() - t0
        print(f"❌ TIMEOUT après {duree:.1f}s — CE TOOL EST LE PROBLÈME")
    except requests.exceptions.ConnectionError:
        print("❌ Connexion refusée — l'API tourne-t-elle ?")


if __name__ == "__main__":
    for nom, prompt in TESTS:
        tester_un_prompt(nom, prompt)
        input("\n[Entrée pour continuer au test suivant, Ctrl+C pour arrêter]")