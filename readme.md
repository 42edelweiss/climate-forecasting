# 🔮 DELPHI - Time Series Oracle

*Prédire l'avenir avec la science des données*

**Delphi** est une application complète d'analyse et de prévision de séries temporelles qui combine :
- **Méthodologie Box-Jenkins** (ARIMA/SARIMA) pour séries univariées - la sagesse classique
- **Deep Learning** (LSTM/GRU) pour séries multivariées - la puissance moderne
- **Interface intuitive** Streamlit - accessible à tous

> *Comme l'oracle de Delphes prédisait l'avenir dans la mythologie grecque, Delphi utilise les mathématiques et l'IA pour dévoiler les tendances cachées dans vos données.*

---

## 🚀 Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer Graphviz (si besoin de visualiser les graphes computationnels)

**Windows:**
```powershell
winget install graphviz
# Puis redémarrer le terminal
```

**Linux:**
```bash
sudo apt-get install graphviz
```

**Mac:**
```bash
brew install graphviz
```

---

## 📊 Utilisation

## 🔮 Lancement de Delphi

### Démarrer l'oracle

```bash
streamlit run delphi.py
```

L'application s'ouvrira automatiquement dans votre navigateur à `http://localhost:8501`

### Générer des données d'exemple

```bash
python generate_example_data.py
```

Cela créera un fichier `exemple_series_temporelles.csv` que vous pourrez charger dans l'application.

---

## 📋 Fonctionnalités

### 🏠 Page Accueil
- Vue d'ensemble de l'application
- Statistiques descriptives des données

### 🔍 Exploration des Données
- Visualisation des séries temporelles
- Statistiques par variable
- Matrice de corrélation
- Distribution des valeurs

### 📦 Box-Jenkins (Univarié)

**Méthodologie complète en 5 étapes :**

1. **Identification** 
   - Tests de stationnarité (ADF)
   - Suggestions automatiques pour la différenciation

2. **Spécification**
   - Graphiques ACF/PACF
   - Détermination des ordres p, d, q

3. **Estimation**
   - Ajustement ARIMA(p,d,q)
   - Ajustement SARIMA(p,d,q)(P,D,Q,s)
   - Résumé statistique complet

4. **Diagnostic**
   - Analyse des résidus
   - Test de Ljung-Box
   - Q-Q plot
   - ACF des résidus

5. **Prévision**
   - Prévisions futures avec intervalles de confiance
   - Métriques de performance (MSE, RMSE, MAE)

### 🤖 Deep Learning (Multivarié)

**Modèles disponibles :**
- **LSTM** - Long Short-Term Memory
- **GRU** - Gated Recurrent Unit
- **VAR** - Vector Autoregression (à venir)

**Fonctionnalités :**
- Configuration flexible des hyperparamètres
- Visualisation de la courbe d'apprentissage
- Prédictions multi-horizons
- Métriques de performance

### 📊 Comparaison des Modèles (à venir)
- Comparaison automatique de tous les modèles
- Tableau de bord des métriques
- Recommandation du meilleur modèle

---

## 📁 Structure des Fichiers

```
.
├── delphi.py                        # 🔮 Application principale DELPHI
├── delphi_branding.py               # 🏛️ Logo, banners et images
├── delphi_style.css                 # 🎨 Style CSS personnalisé (thème grec)
├── requirements.txt                 # 📦 Dépendances Python
├── generate_example_data.py         # 📊 Script pour générer des données exemple
├── README.md                        # 📖 Ce fichier
├── ABOUT.md                         # ℹ️ Philosophie et identité de Delphi
└── exemple_series_temporelles.csv  # 📈 Données exemple (généré)
```

## 🎨 Identité Visuelle

Delphi intègre une identité visuelle inspirée de la Grèce antique :

- **🏛️ Logo ASCII Art** - Temple de Delphes stylisé
- **📸 Images historiques** - Photos des ruines de Delphes (domaine public)
- **🎨 Thème visuel** - Couleurs or et bleu inspirées de l'architecture grecque
- **✨ Banners** - Pour chaque section (Box-Jenkins, Deep Learning, Prévisions)

Les images sont chargées depuis Wikimedia Commons (domaine public) et s'affichent automatiquement sur la page d'accueil.

---

## 📊 Format des Données

L'application accepte des fichiers CSV avec :
- Une colonne de dates (format reconnu par pandas)
- Une ou plusieurs colonnes numériques

**Exemple :**
```csv
Date,Variable_1,Variable_2,Variable_3
2020-01-01,100.5,85.3,50.2
2020-01-02,102.3,87.1,51.0
...
```

---

## 🎯 Cas d'Usage

### Série Univariée (Box-Jenkins)
1. Charger vos données CSV
2. Aller dans "Box-Jenkins (Univarié)"
3. Sélectionner votre variable cible
4. Suivre les 5 étapes de la méthodologie
5. Obtenir des prévisions avec intervalles de confiance

### Séries Multivariées (Deep Learning)
1. Charger vos données CSV avec plusieurs variables
2. Aller dans "Deep Learning (Multivarié)"
3. Sélectionner variable cible et variables explicatives
4. Choisir LSTM ou GRU
5. Configurer les hyperparamètres
6. Entraîner et obtenir les prévisions

---

## 🔧 Paramètres Recommandés

### Box-Jenkins
- **Lookback ACF/PACF:** 30-50 lags pour voir les patterns
- **Différenciation:** Commencer par d=1 si série non-stationnaire
- **SARIMA:** Utiliser si saisonnalité évidente (s=12 mensuel, s=7 hebdomadaire)

### LSTM/GRU
- **Lookback:** 30-60 observations historiques
- **Hidden size:** 64-128 pour commencer
- **Num layers:** 2-3 couches
- **Epochs:** 50-100 (surveiller la convergence)

---

## ⚠️ Notes Importantes

- Pour les séries très longues, l'entraînement LSTM/GRU peut être lent sur CPU
- La différenciation est automatiquement suggérée selon le test ADF
- Les intervalles de confiance ARIMA supposent la normalité des résidus
- Pour de meilleures performances DL, utiliser un GPU si disponible

---

## 📚 Références Théoriques

### Box-Jenkins
- Box, G. E. P., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control

### Deep Learning
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory
- Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder

---

## 🎯 Pourquoi "Delphi" ?

Dans la Grèce antique, l'**Oracle de Delphes** était consulté pour prédire l'avenir. Aujourd'hui, **Delphi** utilise les statistiques modernes et le deep learning pour dévoiler les patterns cachés dans vos données et prédire les tendances futures.

---

## 🤝 Contribution

**Delphi** est conçu pour l'analyse professionnelle de séries temporelles.
Parfait pour :
- 📚 Projets académiques en statistiques/ML
- 💹 Analyse de données financières
- 📊 Prévisions de demande
- 🌍 Analyse de données environnementales
- 🔬 Recherche scientifique

---

## 📧 Support

L'oracle est à votre disposition ! Pour toute question ou suggestion d'amélioration, n'hésitez pas.

**Que Delphi guide vos prévisions ! 🔮📈**