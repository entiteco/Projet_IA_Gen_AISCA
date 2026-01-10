import os
import google.generativeai as genai

# On récupère la clé depuis l'environnement Docker
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("❌ Erreur : Pas de clé API trouvée dans les variables d'environnement.")
    exit()

print(f"✅ Clé trouvée : {api_key[:5]}...*****")

try:
    genai.configure(api_key=api_key)
    
    print("\n🔍 Recherche des modèles disponibles pour 'generateContent'...")
    print("-" * 50)
    
    found = False
    for m in genai.list_models():
        # On ne veut que les modèles qui savent générer du texte
        if 'generateContent' in m.supported_generation_methods:
            print(f"👉 {m.name}")
            found = True
            
    if not found:
        print("⚠️ Aucun modèle compatible trouvé.")
        
    print("-" * 50)

except Exception as e:
    print(f"❌ Erreur critique lors de la connexion : {e}")