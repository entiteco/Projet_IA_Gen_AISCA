### https://aistudio.google.com/app/api-keys

### 🧪 Test 1 : Le "Data Scientist" Idéal (Happy Path)

*Ce test vérifie que le matching simple fonctionne quand tout est aligné.*

* **Texte (Histoire) :** "J'adore les statistiques et les probabilités. Mon but est de créer des modèles prédictifs complexes pour anticiper des comportements futurs. Je code tous les jours en Python."
* **Domaine :** Mathématiques & Recherche
* **Compétences cochées :** Python, SQL, Git
* **Slider Code :** 5/5
* **Slider Maths :** Expert
* **👉 Résultat attendu :**
  * **Métier :** *Data Scientist Senior* ou  *Machine Learning Engineer* .
  * **Score :** Très haut (> 80%).
  * **Gemini :** Doit dire un truc du genre : *"Votre profil technique est parfait, foncez !"*

---

### 🧪 Test 2 : Le "Data Analyst" (Business & Visuel)

*Ce test vérifie que l'IA distingue bien l'analyse (Viz) de la science (Maths).*

* **Texte (Histoire) :** "Je ne suis pas un grand codeur, mais j'aime faire parler les chiffres pour aider les managers à décider. J'adore créer des tableaux de bord interactifs et visuels pour raconter une histoire avec la donnée."
* **Domaine :** Analyse & Business
* **Compétences cochées :** Excel, PowerBI, SQL
* **Slider Code :** 2/5
* **Slider Maths :** Intermédiaire
* **👉 Résultat attendu :**
  * **Métier :** *Data Analyst* ou  *Consultant BI* .
  * **Gemini :** Doit mettre l'accent sur votre capacité de "Storytelling" et l'utilisation d'outils comme PowerBI.

---

### 🧪 Test 3 : Le "Crash Test" (Le fan d'IA nul en maths)

*Ce test est CRITIQUE. Il vérifie que votre règle de pénalité (-60%) fonctionne.*

* **Texte (Histoire) :** "Je rêve de travailler dans l'Intelligence Artificielle et de créer des robots comme ChatGPT. C'est ma passion absolue."
* **Domaine :** Peu importe
* **Compétences cochées :** (Aucune ou juste Excel)
* **Slider Code :** 1/5
* **Slider Maths :** **Débutant** (⚠️ Le piège est ici)
* **👉 Résultat attendu :**
  * **Métier :** *Data Scientist* doit **disparaître** du top (ou avoir un score très bas rouge).
  * **Raison affichée :** Vous devez voir le message rouge :  **⛔ Bloqué par le niveau Maths (-60%)** .
  * **Recommandation :** L'IA devrait peut-être proposer *Data Steward* ou *Product Owner* (des métiers moins techniques).

---

### 🧪 Test 4 : L'Architecte Cloud (L'Ingénieur pur)

*Ce test vérifie le vocabulaire technique (Infrastructure).*

* **Texte (Histoire) :** "Je n'aime pas trop les maths et les analyses. Ce qui me plaît, c'est de construire des tuyaux solides pour déplacer la donnée. J'aime automatiser les serveurs et gérer le cloud."
* **Domaine :** Infrastructure & Cloud
* **Compétences cochées :** AWS, Docker, Python
* **Slider Code :** 4/5
* **Slider Maths :** Notions
* **👉 Résultat attendu :**
  * **Métier :** *Data Engineer* ou  *Cloud Data Engineer* .
  * **Gemini :** Doit valider le côté "Ops" et "Construction" plutôt que l'analyse.



 **OUI, SBERT est utilisé** , et il est le moteur central de votre application. Sans lui, aucune compréhension du sens n'est possible.

Voici l'explication détaillée du fonctionnement de votre `nlp_engine.py` (version académique mise à jour), étape par étape.

### 1. Le Cerveau : SBERT (`Sentence-BERT`)

```python
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

* **C'est quoi ?** Vous utilisez le modèle `paraphrase-multilingual-MiniLM-L12-v2`.
* **Son rôle :** C'est un  **traducteur** . Il ne traduit pas du français vers l'anglais, mais du  **Français vers les Mathématiques** .
* **Le Vectorisation (Embedding) :** SBERT prend une phrase (ex:  *"J'aime prédire l'avenir avec des stats"* ) et la transforme en une liste de 384 nombres (un vecteur). Ces nombres représentent le **sens** de la phrase.
* **La magie :** Deux phrases écrites différemment mais avec le même sens (ex: *"Je fais du Machine Learning"* et  *"Je crée des modèles prédictifs"* ) auront des vecteurs très proches mathématiquement.

### 2. L'Étape Intermédiaire : Les Blocs de Compétences

C'est la grande différence avec votre ancienne version. Au lieu de comparer directement le candidat au métier, on passe par des concepts abstraits (les Blocs).

* **Le code :** `self.block_embeddings`
* **L'action :** Au démarrage, le système calcule le "vecteur de référence" pour chaque bloc (Data Analysis, NLP, Soft Skills...) en utilisant leur description textuelle.

### 3. Le Calcul des Scores par Bloc (**S**i)

Quand l'utilisateur envoie son texte, la fonction `calculate_block_scores(user_text)` s'active :

1. Elle transforme le texte utilisateur en vecteur (grâce à SBERT).
2. Elle mesure la **Similarité Cosinus** (l'angle) entre le vecteur utilisateur et le vecteur de chaque Bloc.

**Résultat concret :**

* Utilisateur : *"Je code en Python pour nettoyer des fichiers CSV."*
* SBERT compare avec le Bloc "Data Analysis" -> Angle très faible -> Score 0.85 (85%)
* SBERT compare avec le Bloc "NLP" -> Angle grand -> Score 0.20 (20%)

### 4. La Recommandation Pondérée (Formule **W**i)

C'est ici que l'exigence du cahier des charges est respectée.
La fonction `find_top_matches` ne fait plus une comparaison directe. Elle applique la formule :

**S**core**=**∑**P**o**i**d**s**∑**(**P**o**i**d**s**×**S**core**Bl**oc**)

* **Logique :** Pour un poste de "Data Scientist", le code (`get_job_weights`) dit que le bloc "Machine Learning" pèse plus lourd (Poids = 2.0) que le bloc "Soft Skills" (Poids = 1.0).
* L'algorithme combine donc les scores SBERT de l'utilisateur avec l'importance de chaque compétence pour le métier visé.

### 5. Les Règles Métiers (Le "Garde-Fou")

Enfin, la méthode `apply_business_rules` intervient après le calcul SBERT.
C'est une couche de **logique pure (If/Else)** qui vient corriger l'IA.

* **Pourquoi ?** SBERT peut trouver que "J'adore la Data Science mais je suis nul en maths" ressemble sémantiquement à "Data Scientist".
* **Le filtre :** Votre code vérifie les sliders (Niveau Maths/Code). Si le niveau est trop bas pour un métier exigeant (ex: Scientist), il applique un **MALUS brutal** (-60%) au score final.

---

### Résumé pour le Jury

Si on vous demande "Comment ça marche ?", répondez ceci :

1. **Ingestion :** On utilise **SBERT** pour vectoriser le texte du candidat.
2. **Matching Sémantique :** On compare ce vecteur à 5 **Blocs de Compétences** clés (Data Analysis, ML, etc.) pour obtenir un profil technique (ex: 80% Dev, 20% IA).
3. **Algorithme de Pondération :** On projette ce profil sur les métiers. Chaque métier a des "poids" différents (un Data Engineer a besoin de plus de code qu'un Analyst).
4. **Filtrage Hybride :** On applique des règles strictes (niveau maths/code) pour pénaliser les profils qui matchent sémantiquement mais n'ont pas le niveau technique requis.
