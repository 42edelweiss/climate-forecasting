# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ✅ Branding
from delphi_branding import apply_branding, footer, banner_boxjenkins, banner_deeplearning


# =============================================================================
# TXT -> DF (dioxcar-like)
# =============================================================================
def parse_dioxcar_txt(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse un fichier TXT du style:
        "x"
        "1" 315.42
        "2" 316.32
        ...
    Retour:
        DataFrame index datetime (mensuel) + colonne 'x'
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # ignore header "x"
        if line.replace('"', "").strip().lower() == "x":
            continue

        m = re.match(r'^"?(?P<t>\d+)"?\s+(?P<val>-?\d+(?:\.\d+)?)$', line)
        if m:
            t = int(m.group("t"))
            val = float(m.group("val"))
            rows.append((t, val))

    if not rows:
        raise ValueError("Format TXT non reconnu. Exemple attendu: \"1\" 315.42")

    df_txt = pd.DataFrame(rows, columns=["t", "x"])

    # ✅ index datetime mensuel (MS) - parfait pour CO2 type dioxcar
    df_txt["date"] = pd.date_range("1959-01-01", periods=len(df_txt), freq="MS")
    df_txt = df_txt.set_index("date").drop(columns=["t"]).sort_index()

    return df_txt


# =============================================================================
# Helpers
# =============================================================================
def infer_freq(index: pd.DatetimeIndex) -> str:
    """Essaie d'inférer une fréquence pandas; fallback sur 'D'."""
    try:
        freq = pd.infer_freq(index)
        return freq if freq is not None else "D"
    except Exception:
        return "D"


def safe_numeric_cols(dataframe: pd.DataFrame):
    return dataframe.select_dtypes(include=[np.number]).columns.tolist()


# =============================================================================
# CONFIG STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Delphi - Time Series Oracle",
    page_icon="🔮",
    layout="wide",
)

# ✅ IMPORTANT: branding juste après set_page_config
apply_branding(show_logo=True, show_temple=False)

st.title("🔮 DELPHI - Time Series Oracle")
st.markdown("*Prédire l'avenir avec Box-Jenkins & Deep Learning*")
st.markdown("---")


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choisir le mode",
    [
        "Accueil",
        "Exploration des Données",
        "Box-Jenkins (Univarié)",
        "Deep Learning (Multivarié)",
        "Comparaison des Modèles",
    ],
)

st.sidebar.title("Import des Données")
uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV ou TXT", type=["csv", "txt"])

df = None

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()

        # --------------------
        # CSV
        # --------------------
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ CSV chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")

            # Sélection colonne date
            date_column = st.sidebar.selectbox("Colonne de date", df.columns)
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.dropna(subset=[date_column])
            df = df.set_index(date_column).sort_index()

        # --------------------
        # TXT (dioxcar-like)
        # --------------------
        elif filename.endswith(".txt"):
            df = parse_dioxcar_txt(uploaded_file.getvalue())
            st.sidebar.success(f"✅ TXT converti: {df.shape[0]} lignes, {df.shape[1]} colonne(s)")

            # ✅ bouton download CSV
            csv_bytes = df.reset_index().to_csv(index=False).encode("utf-8")
            st.sidebar.download_button(
                "⬇️ Télécharger le CSV converti",
                data=csv_bytes,
                file_name="dioxcar_converted.csv",
                mime="text/csv",
            )

        else:
            st.sidebar.error("❌ Format non supporté.")
            st.stop()

    except Exception as e:
        st.sidebar.error(f"❌ Impossible de lire le fichier: {e}")
        st.stop()
else:
    st.info("👈 Veuillez charger un fichier CSV ou TXT pour commencer")
    st.stop()


# =============================================================================
# PAGE: ACCUEIL
# =============================================================================
if app_mode == "Accueil":
    st.header("🔮 Bienvenue dans DELPHI")
    st.subheader("*L'Oracle des Séries Temporelles*")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Pour les séries univariées")
        st.markdown(
            """
            **Méthodologie Box-Jenkins complète:**
            1. **Identification** - Tests de stationnarité (ADF)
            2. **Spécification** - Analyse ACF/PACF pour déterminer p, d, q
            3. **Estimation** - Ajustement du modèle ARIMA/SARIMA
            4. **Diagnostic** - Analyse des résidus (Ljung-Box)
            5. **Prévision** - Prédictions futures
            """
        )

    with col2:
        st.subheader("🤖 Pour les séries multivariées")
        st.markdown(
            """
            **Méthodes de Deep Learning:**
            - **LSTM**
            - **GRU**
            - (VAR classique à venir)

            Comparaison automatique des performances (à venir) !
            """
        )

    st.subheader("📋 Aperçu des données")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📊 Statistiques descriptives")
    st.dataframe(df.describe(include="all"), use_container_width=True)


# =============================================================================
# PAGE: EXPLORATION DES DONNÉES
# =============================================================================
elif app_mode == "Exploration des Données":
    st.header("🔍 Exploration des Données")

    numeric_cols = safe_numeric_cols(df)
    if not numeric_cols:
        st.warning("Aucune colonne numérique détectée.")
        st.stop()

    selected_cols = st.multiselect(
        "Sélectionner les colonnes à visualiser",
        numeric_cols,
        default=numeric_cols[:1],
    )

    if not selected_cols:
        st.warning("Sélectionne au moins une colonne.")
        st.stop()

    st.subheader("📈 Visualisation des Séries Temporelles")
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in selected_cols:
        ax.plot(df.index, df[col], label=col, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur")
    ax.set_title("Séries Temporelles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("📊 Statistiques par Variable")
    st.dataframe(df[selected_cols].describe(), use_container_width=True)

    if len(selected_cols) > 1:
        st.subheader("🔗 Matrice de Corrélation")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

    st.subheader("📊 Distribution des Valeurs")
    fig, axes = plt.subplots(1, len(selected_cols), figsize=(6 * len(selected_cols), 4))
    if len(selected_cols) == 1:
        axes = [axes]
    for i, col in enumerate(selected_cols):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        axes[i].set_title(f"Distribution - {col}")
        axes[i].set_xlabel("Valeur")
        axes[i].set_ylabel("Fréquence")
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# =============================================================================
# PAGE: BOX-JENKINS (UNIVARIÉ)
# =============================================================================
elif app_mode == "Box-Jenkins (Univarié)":
    st.header("📦 Méthodologie Box-Jenkins")
    banner_boxjenkins()

    numeric_cols = safe_numeric_cols(df)
    if not numeric_cols:
        st.warning("Aucune colonne numérique détectée.")
        st.stop()

    target_col = st.selectbox("Sélectionner la variable cible", numeric_cols)

    series = df[target_col].dropna()
    if len(series) < 20:
        st.warning("Série trop courte (moins de ~20 points). Ajoute plus de données.")
        st.stop()

    st.subheader("1️⃣ Identification - Tests de Stationnarité")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Test ADF (Augmented Dickey-Fuller)**")
        try:
            adf_result = adfuller(series)
            st.write(f"- **Statistique ADF:** {adf_result[0]:.4f}")
            st.write(f"- **p-value:** {adf_result[1]:.4f}")
            st.write("- **Valeurs critiques:**")
            for key, value in adf_result[4].items():
                st.write(f"  - {key}: {value:.4f}")

            d_suggested = 0 if adf_result[1] < 0.05 else 1
            if d_suggested == 0:
                st.success("✅ Série stationnaire (p < 0.05)")
            else:
                st.warning("⚠️ Série non-stationnaire (p >= 0.05) - Différenciation requise")
        except Exception as e:
            st.error(f"Erreur ADF: {e}")
            d_suggested = 1

    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(series.index, series.values)
        ax.set_title(f"Série Temporelle - {target_col}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Valeur")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    diff_order = st.slider("Ordre de différenciation (d)", 0, 2, d_suggested)

    if diff_order > 0:
        series_for_acf = series.diff(diff_order).dropna()
        st.write(f"Série différenciée (d={diff_order})")

        try:
            adf_diff = adfuller(series_for_acf)
            st.write(f"- **ADF après différenciation:** {adf_diff[0]:.4f}")
            st.write(f"- **p-value:** {adf_diff[1]:.4f}")
            if adf_diff[1] < 0.05:
                st.success("✅ Série maintenant stationnaire")
            else:
                st.warning("⚠️ Encore non-stationnaire — essaye d=2 ou SARIMA.")
        except Exception as e:
            st.error(f"Erreur ADF (diff): {e}")
    else:
        series_for_acf = series

    st.subheader("2️⃣ Spécification - Analyse ACF/PACF")
    max_lags = st.slider("Nombre de lags à afficher", 10, 50, 30)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(series_for_acf, lags=max_lags, ax=ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(series_for_acf, lags=max_lags, ax=ax, method="ywm")
        st.pyplot(fig)

    st.subheader("3️⃣ Estimation - Ajustement du Modèle")

    c1, c2, c3 = st.columns(3)
    with c1:
        p = st.number_input("Ordre AR (p)", 0, 10, 1)
    with c2:
        d = st.number_input("Ordre de différenciation (d)", 0, 2, diff_order)
    with c3:
        q = st.number_input("Ordre MA (q)", 0, 10, 1)

    use_seasonal = st.checkbox("Utiliser SARIMA (composante saisonnière)")
    if use_seasonal:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            P = st.number_input("P (AR saisonnier)", 0, 5, 1)
        with c2:
            D = st.number_input("D (diff saisonnière)", 0, 2, 1)
        with c3:
            Q = st.number_input("Q (MA saisonnier)", 0, 5, 1)
        with c4:
            s = st.number_input("Période (s)", 1, 365, 12)
        seasonal_order = (P, D, Q, s)
    else:
        seasonal_order = (0, 0, 0, 0)

    if st.button("🚀 Ajuster le Modèle"):
        with st.spinner("Ajustement du modèle en cours..."):
            try:
                if use_seasonal:
                    model = SARIMAX(
                        series,
                        order=(p, d, q),
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    model_name = f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"
                else:
                    model = ARIMA(series, order=(p, d, q))
                    model_name = f"ARIMA({p},{d},{q})"

                results = model.fit()
                st.success(f"✅ Modèle {model_name} ajusté avec succès!")
                st.text(results.summary())

                st.session_state["model"] = results
                st.session_state["model_name"] = model_name
                st.session_state["series"] = series

            except Exception as e:
                st.error(f"❌ Erreur lors de l'ajustement: {e}")

    if "model" in st.session_state:
        st.subheader("4️⃣ Diagnostic - Analyse des Résidus")
        results = st.session_state["model"]
        residuals = results.resid.dropna()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(residuals)
        ax.axhline(y=0, linestyle="--", alpha=0.5)
        ax.set_title("Résidus")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Test de Ljung-Box")
        lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
        st.dataframe(lb, use_container_width=True)


# =============================================================================
# PAGE: DEEP LEARNING (MULTIVARIÉ)
# =============================================================================
elif app_mode == "Deep Learning (Multivarié)":
    st.header("🤖 Modèles Deep Learning pour Séries Multivariées")
    banner_deeplearning()

    numeric_cols = safe_numeric_cols(df)
    if len(numeric_cols) < 2:
        st.warning("Il faut au moins 2 colonnes numériques pour du multivarié.")
        st.stop()

    st.subheader("Sélection des Variables")
    target_col = st.selectbox("Variable cible à prédire", numeric_cols)

    feature_cols = st.multiselect(
        "Variables explicatives",
        [c for c in numeric_cols if c != target_col],
        default=[c for c in numeric_cols if c != target_col][:3],
    )

    if not feature_cols:
        st.warning("Veuillez sélectionner au moins une variable explicative")
        st.stop()

    st.subheader("⚙️ Configuration du Modèle")
    c1, c2, c3 = st.columns(3)
    with c1:
        lookback = st.slider("Fenêtre temporelle (lookback)", 5, 100, 30)
    with c2:
        train_split = st.slider("% données d'entraînement", 50, 90, 80) / 100
    with c3:
        forecast_horizon = st.slider("Horizon de prévision", 1, 30, 10)

    model_type = st.selectbox("Sélectionner le modèle", ["LSTM", "GRU"])

    c1, c2, c3 = st.columns(3)
    with c1:
        hidden_size = st.slider("Taille cachée", 16, 256, 64)
    with c2:
        num_layers = st.slider("Nombre de couches", 1, 5, 2)
    with c3:
        epochs = st.slider("Époques", 10, 200, 50)

    all_cols = [target_col] + feature_cols
    data_df = df[all_cols].dropna()

    if len(data_df) < (lookback + forecast_horizon + 10):
        st.warning("Pas assez de données après nettoyage pour créer des séquences.")
        st.stop()

    data = data_df.values.astype(np.float32)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    def create_sequences(arr, lookback, horizon):
        X, y = [], []
        for i in range(len(arr) - lookback - horizon + 1):
            X.append(arr[i : i + lookback, :])
            y.append(arr[i + lookback : i + lookback + horizon, 0])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X, y = create_sequences(data_scaled, lookback, forecast_horizon)

    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    input_size = len(all_cols)
    output_size = forecast_horizon

    if st.button(f"🚀 Entraîner le modèle {model_type}"):
        with st.spinner(f"Entraînement du modèle {model_type} en cours..."):
            model = (
                LSTMModel(input_size, hidden_size, num_layers, output_size)
                if model_type == "LSTM"
                else GRUModel(input_size, hidden_size, num_layers, output_size)
            )

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            progress_bar = st.progress(0.0)
            loss_history = []

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()

                loss_history.append(float(loss.item()))
                progress_bar.progress((epoch + 1) / epochs)

            st.success("✅ Modèle entraîné!")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(loss_history)
            ax.set_title("Loss")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test_t).cpu().numpy()

            n_features = len(all_cols)
            y_test_flat = y_test.reshape(-1, 1)
            y_pred_flat = y_pred_test.reshape(-1, 1)

            def inverse_first_col(y_flat):
                tmp = np.zeros((len(y_flat), n_features), dtype=np.float32)
                tmp[:, 0] = y_flat[:, 0]
                inv = scaler.inverse_transform(tmp)
                return inv[:, 0]

            y_test_actual = inverse_first_col(y_test_flat)
            y_pred_actual = inverse_first_col(y_pred_flat)

            mse = mean_squared_error(y_test_actual, y_pred_actual)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_test_actual, y_pred_actual)

            c1, c2, c3 = st.columns(3)
            c1.metric("MSE", f"{mse:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("MAE", f"{mae:.4f}")


# =============================================================================
# PAGE: COMPARAISON
# =============================================================================
elif app_mode == "Comparaison des Modèles":
    st.header("📊 Comparaison des Modèles")
    st.info("Cette fonctionnalité sera disponible après avoir entraîné plusieurs modèles")


# ✅ Footer à la fin (hors if/elif)
footer()
