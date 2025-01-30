import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump, load, parallel_backend

# Optional: XGBoost / LightGBM (falls installiert)
try:
    from xgboost import XGBRegressor

    xgboost_available = True
except ImportError:
    xgboost_available = False

try:
    from lightgbm import LGBMRegressor

    lightgbm_available = True
except ImportError:
    lightgbm_available = False


def load_and_clean_data(file_paths):
    """
    Lädt und bereinigt mehrere IMDb-Datensätze aus TSV-Dateien.

    Schritte:
    - Liest die Dateien aus den übergebenen Pfaden mit sep='\t'.
    - Wandelt '\\N' zu NaN.
    - Entfernt vollständig leere Zeilen/Spalten.
    - Konvertiert numerische Spalten, wo es geht.
    - Speichert jede bereinigte Tabelle in 'cleaned_data_single/' ab.
    - Gibt ein Dictionary mit den bereinigten DataFrames zurück.
    """
    data = {}
    if not os.path.exists("cleaned_data_single"):
        os.makedirs("cleaned_data_single")

    for name, path in file_paths.items():
        print(f"[load_and_clean_data] Lese Datei '{name}' von: {path}")
        df = pd.read_csv(path, sep='\t', dtype=str, na_values=['\\N'])

        if name == "title_akas" and "titleId" in df.columns:
            print("[load_and_clean_data] 'title_akas' hat 'titleId' -> rename zu 'tconst'")
            df.rename(columns={"titleId": "tconst"}, inplace=True)

        # Entferne rein leere Zeilen und Spalten
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        # Typkonvertierung (versuche jede object-Spalte in numerisch zu wandeln)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

        # Speichere bereinigte Tabelle ab
        out_path = os.path.join("cleaned_data_single", f"{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[load_and_clean_data] '{name}' gespeichert unter: {out_path} "
              f"(Zeilen={len(df)}, Spalten={len(df.columns)})")

        data[name] = df
    return data


def prepare_single():
    """
    Erzeugt einen aggregierten (Master-)Datensatz aus mehreren IMDb-Dateien.

    Hauptschritte:
      1. Laden & Bereinigen der jeweiligen Einzel-TSVs via load_and_clean_data().
      2. Zusammenführen von 'title_basics' und 'title_ratings' (nur movies/tvSeries).
      3. Hashen von:
         - titleType (als titleType_hashed)
         - directors, writers (directors_hashed, writers_hashed)
         - aggregierte Kategorien (categories_hashed)
      4. Aggregation von Sprachen & Originaltiteln aus title_akas.
      5. Entfernen von End- und Hilfsspalten (z. B. endYear, ordering, title, region, language,
         types, attributes, isOriginalTitle, runtimeMinutes).
      6. Filtern: nur numVotes > 100.
      7. Speichern des fertigen Master-Datensatzes als CSV in 'master_features/master_dataset.csv'.
      8. Split in Train- und Testset (80/20), Ablage in 'master_train_data' bzw. 'master_test_data'.
    """
    print("[prepare_single] Starte das Erstellen des Master-Datasets...")

    file_paths = {
        "title_akas": "./data/title.akas.tsv",
        "title_principals": "./data/title.principals.tsv",
        "title_basics": "./data/title.basics.tsv",
        "title_ratings": "./data/title.ratings.tsv",
        "title_episodes": "./data/title.episode.tsv",
        "title_crew": "./data/title.crew.tsv",
        "name_basics": "./data/name.basics.tsv",
    }

    # Laden und Bereinigen
    raw_data = load_and_clean_data(file_paths)

    # Mergen: title_basics + title_ratings
    print("[prepare_single] Merging 'title_basics' und 'title_ratings'...")
    merged = pd.merge(raw_data["title_basics"], raw_data["title_ratings"], on="tconst", how="inner")

    # Hash: titleType => titleType_hashed
    print("[prepare_single] Hashing 'titleType'...")
    merged["titleType_hashed"] = merged["titleType"].astype(str).apply(lambda x: hash(x) % 100000)
    if "titleType" in merged.columns:
        merged.drop(columns=["titleType"], inplace=True, errors="ignore")

    # Merge title_crew => Hash von directors, writers
    if "title_crew" in raw_data:
        print("[prepare_single] Merging 'title_crew'...")
        merged = pd.merge(merged, raw_data["title_crew"], on="tconst", how="left")
        merged["directors_hashed"] = merged["directors"].astype(str).apply(lambda x: hash(x) % 100000)
        merged["writers_hashed"] = merged["writers"].astype(str).apply(lambda x: hash(x) % 100000)
        merged.drop(columns=["directors", "writers"], inplace=True, errors="ignore")

    # Kategorien aus title_principals => categories_hashed
    if "title_principals" in raw_data:
        print("[prepare_single] Aggregiere Kategorien aus 'title_principals'...")
        principals = raw_data["title_principals"].copy()
        principals["category"] = principals["category"].astype(str)
        cat_agg = principals.groupby("tconst")["category"] \
            .apply(lambda x: "|".join(sorted(set(x)))) \
            .reset_index(name="all_categories_str")
        cat_agg["categories_hashed"] = cat_agg["all_categories_str"].apply(lambda s: hash(s) % 100000)
        merged = pd.merge(merged, cat_agg[["tconst", "categories_hashed"]], how="left", on="tconst")
        merged["categories_hashed"] = merged["categories_hashed"].fillna(0)

    # Daten aus title_akas => Sprachen & isOriginal
    if "title_akas" in raw_data:
        print("[prepare_single] Aggregiere Sprache & Originaltitel aus 'title_akas'...")
        akas = raw_data["title_akas"].copy()
        akas["language"] = akas["language"].astype(str)
        akas_grp = akas.groupby("tconst").agg(
            num_languages=("language", "nunique"),
            original_count=("isOriginalTitle", lambda x: sum(x == 1))
        ).reset_index()

        merged = pd.merge(merged, akas_grp, on="tconst", how="left")
        merged["num_languages"] = merged["num_languages"].fillna(0)
        merged["has_originalTitle_aka"] = (merged["original_count"] > 0).astype(int)
        if "original_count" in merged.columns:
            merged.drop(columns=["original_count"], inplace=True)

    # Unerwünschte Spalten entfernen
    drop_cols = [
        "endYear", "ordering", "title", "region", "language",
        "types", "attributes", "isOriginalTitle", "runtimeMinutes", "has_originalTitle_aka", "num_languages", "genres", "originalTitle", "primaryTitle", "tconst"
    ]
    for c in drop_cols:
        if c in merged.columns:
            merged.drop(columns=[c], inplace=True)
            print(f"[prepare_single] Spalte '{c}' entfernt.")

    # Filter: numVotes > 50
    merged = merged[merged["numVotes"] > 50].copy()
    print(f"[prepare_single] Nach Filter (numVotes>100): {len(merged)} Zeilen")

    # Filter: startYear nach 1900
    merged = merged[merged["startYear"] > 1900].copy()
    print(f"[prepare_single] Nach Filter (numVotes>100): {len(merged)} Zeilen")

    # Speichere finalen Master-Datensatz
    if not os.path.exists("master_features"):
        os.makedirs("master_features")
    master_out = os.path.join("master_features", "master_dataset.csv")
    merged.to_csv(master_out, index=False)
    print(f"[prepare_single] Master-Dataset gespeichert unter: {master_out}")

    # Train-/Test-Split
    train_df, test_df = train_test_split(merged, test_size=0.2, random_state=42)
    if not os.path.exists("master_train_data"):
        os.makedirs("master_train_data")
    if not os.path.exists("master_test_data"):
        os.makedirs("master_test_data")

    train_path = os.path.join("master_train_data", "master_train.csv")
    test_path = os.path.join("master_test_data", "master_test.csv")
    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)
    print(f"[prepare_single] Train-Set: {train_path} (Zeilen={len(train_df)})")
    print(f"[prepare_single] Test-Set:  {test_path} (Zeilen={len(test_df)})")


def train_single_data(partial_size=10000, retrain=True):
    """
    Trainiert ein Machine-Learning-Modell auf dem aggregierten Master-Datensatz.

    Schritte:
    - Liest 'master_train_data/master_train.csv'.
    - Nimmt verschiedene Modellkandidaten (GridSearchCV) und wählt den besten.
    - Speichert das beste Modell in 'master_models/master_model.joblib'.
    - Beschränkt Training optional auf 'partial_size' Datensätze.
    - Gibt Debug-Infos über den Prozess aus.
    """
    print("[train_single_data] Starte das Training für den Master-Datensatz...")
    train_dir = "master_train_data"
    model_dir = "master_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_path = os.path.join(train_dir, "master_train.csv")
    if not os.path.exists(train_path):
        print(f"[train_single_data] Trainingsdatei existiert nicht: {train_path}")
        return

    df_train = pd.read_csv(train_path, sep="\t")
    if "averageRating" not in df_train.columns or df_train.empty:
        print("[train_single_data] Trainingsdatensatz ist leer oder hat keine Spalte 'averageRating'.")
        return

    model_path = os.path.join(model_dir, "master_model.joblib")
    if os.path.exists(model_path) and not retrain:
        print("[train_single_data] Modell existiert bereits, retrain=False => Abbruch.")
        return

    print(f"[train_single_data] Datensatz für Training geladen: {train_path} (Zeilen={len(df_train)})")

    # Evtl. nur Teilmenge nehmen
    if len(df_train) > partial_size:
        print(f"[train_single_data] Reduziere auf {partial_size} Zeilen (Stichprobe)...")
        df_train = df_train.sample(n=partial_size, random_state=42).copy()

    # Features und Ziel trennen
    drop_cols = ["tconst", "averageRating"]
    X_train = df_train.drop(columns=drop_cols, errors="ignore")
    y_train = df_train["averageRating"]

    print(f"[train_single_data] Features: {list(X_train.columns)}")

    # Preprocessing: numerisch vs. kategorisch
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ], remainder="drop")

    # Modellkandidaten
    model_candidates = {
        "linear_reg": (
            LinearRegression(),
            {
                "regressor__fit_intercept": [True, False]
            }
        ),
        "ridge": (
            Ridge(random_state=42),
            {
                "regressor__alpha": [0.5, 1.0, 2.0],
                "regressor__fit_intercept": [True, False]
            }
        ),
        "lasso": (
            Lasso(random_state=42),
            {
                "regressor__alpha": [0.1, 1.0, 2.0],
                "regressor__fit_intercept": [True, False]
            }
        ),
        "decision_tree": (
            DecisionTreeRegressor(random_state=42),
            {
                "regressor__max_depth": [None, 10, 20],
                "regressor__min_samples_split": [2, 5]
            }
        ),
        "knn": (
            KNeighborsRegressor(),
            {
                "regressor__n_neighbors": [3, 5, 10],
                "regressor__weights": ["uniform", "distance"]
            }
        ),
        "svr": (
            SVR(),
            {
                "regressor__kernel": ["rbf", "linear"],
                "regressor__C": [1.0, 10.0]
            }
        ),
        "random_forest": (
            RandomForestRegressor(random_state=42),
            {
                "regressor__n_estimators": [50, 100],
                "regressor__max_depth": [None, 10]
            }
        ),
        "hist_gb": (
            HistGradientBoostingRegressor(random_state=42),
            {
                "regressor__learning_rate": [0.01, 0.05],
                "regressor__max_depth": [5, 10],
                "regressor__max_iter": [100, 200]
            }
        )
    }

    # Evtl. XGBoost/LightGBM ergänzen
    global xgboost_available, lightgbm_available
    if xgboost_available:
        from xgboost import XGBRegressor
        model_candidates["xgboost"] = (
            XGBRegressor(random_state=42, use_label_encoder=False, eval_metric="rmse"),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [3, 6],
                "regressor__learning_rate": [0.01, 0.05]
            }
        )
    if lightgbm_available:
        from lightgbm import LGBMRegressor
        model_candidates["lightgbm"] = (
            LGBMRegressor(random_state=42),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [-1, 10],
                "regressor__learning_rate": [0.01, 0.05]
            }
        )

    best_score = -9999
    best_pipeline = None

    print("[train_single_data] Starte GridSearch über alle Modellkandidaten...")
    for m_name, (model_obj, params) in model_candidates.items():
        print(f"  -> Teste Modell '{m_name}'...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model_obj)
        ])

        param_grid = [params]

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="r2",
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        with parallel_backend("threading"):
            grid_search.fit(X_train, y_train)

        c_score = grid_search.best_score_
        c_est = grid_search.best_estimator_

        print(f"     Bester Score (CV) = {c_score:.4f}, beste Params = {grid_search.best_params_}")

        if c_score > best_score:
            best_score = c_score
            best_pipeline = c_est

    if best_pipeline is None:
        print("[train_single_data] Konnte kein bestes Modell ermitteln.")
        return

    print(f"[train_single_data] Bestes Modell hat Score (CV)={best_score:.4f}. Fitte final auf gesamten (Teil-)Satz...")
    best_pipeline.fit(X_train, y_train)
    final_score = best_pipeline.score(X_train, y_train)

    # Speichere das Modell
    dump(best_pipeline, model_path)
    print(f"[train_single_data] Train-Score (R^2)={final_score:.4f}, Modell gespeichert als {model_path}")


def explore_single_data():
    """
    Erstellt einfache Visualisierungen für den Master-Test-Datensatz.

    Schritte:
    - Lädt 'master_test_data/master_test.csv'.
    - Erstellt Histogramm der averageRating.
    - Für jede Spalte: numerisch => Scatterplot vs. averageRating;
                       kategorisch => Boxplot vs. averageRating (sofern <=30 unique values).
    - Speichert Plots in 'master_exploration/'.
    """
    print("[explore_single_data] Starte Exploration des Master-Testdatensatzes...")
    out_dir = "master_exploration"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_path = os.path.join("master_test_data", "master_test.csv")
    if not os.path.exists(test_path):
        print(f"[explore_single_data] Testdatei nicht vorhanden: {test_path}")
        return

    df_test = pd.read_csv(test_path, sep="\t")
    if "averageRating" not in df_test.columns or df_test.empty:
        print("[explore_single_data] Keine 'averageRating'-Spalte oder Datensatz ist leer.")
        return

    plt.figure(figsize=(6, 4))
    df_test["averageRating"].hist(bins=30, alpha=0.7)
    plt.title("Master: averageRating (Test)")
    plt.xlabel("Rating")
    plt.ylabel("Häufigkeit")
    plt.savefig(os.path.join(out_dir, "master_hist.png"))
    plt.close()
    print("[explore_single_data] -> Histogramm averageRating erstellt.")

    for col in df_test.columns:
        if col == "averageRating":
            continue

        # Scatterplot für numerische Spalten
        if pd.api.types.is_numeric_dtype(df_test[col]):
            df_plot = df_test[[col, "averageRating"]].dropna()
            if df_plot.empty:
                continue
            plt.figure(figsize=(6, 4))
            plt.scatter(df_plot[col], df_plot["averageRating"], alpha=0.3)
            plt.title(f"{col} vs. averageRating")
            plt.xlabel(col)
            plt.ylabel("averageRating")
            plt.savefig(os.path.join(out_dir, f"master_{col}_scatter.png"))
            plt.close()

        # Boxplot für kategorische Spalten (nicht zu viele Kategorien)
        else:
            if df_test[col].nunique() > 30:
                continue
            df_plot = df_test[[col, "averageRating"]].dropna()
            if df_plot.empty:
                continue
            plt.figure(figsize=(8, 4))
            df_plot.boxplot(by=col, column="averageRating", grid=False, rot=90)
            plt.title(f"{col} vs. averageRating")
            plt.suptitle("")  # entfernt den Default-Titel
            plt.ylabel("averageRating")
            plt.savefig(os.path.join(out_dir, f"master_{col}_boxplot.png"))
            plt.close()

    print("[explore_single_data] -> Exploration abgeschlossen.")


def use_single_data(input_data):
    """
    Lädt das trainierte Master-Modell und macht Vorhersagen auf neuen Eingabedaten.

    - input_data kann eine Liste von Dicts oder ein DataFrame sein.
    - Gibt ein Array mit den Vorhersagewerten zurück.
    """
    print("[use_single_data] Lade Modell und führe Vorhersagen durch...")
    model_path = os.path.join("master_models", "master_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden unter: {model_path}")

    pipeline = load(model_path)

    if isinstance(input_data, list):
        df_input = pd.DataFrame(input_data)
    else:
        df_input = input_data.copy()

    preds = pipeline.predict(df_input)
    print(f"[use_single_data] Vorhersagen: {preds}")
    return preds


def main():
    """
    Hauptablauf für den Master-Datensatz:
      1) prepare_single() -> Erstellen & Abspeichern des Master-Datensatzes + Train/Test
      2) train_single_data() -> Trainiert ein Modell (GridSearch) und speichert es
      3) explore_single_data() -> Erstellt einfache Plots auf dem Test-Datensatz
      4) Beispiel-Vorhersage mit use_single_data()
    """
    # 1) Master-Daten vorbereiten
    prepare_single()

    # 2) Modell trainieren
    train_single_data(partial_size=10000, retrain=True)

    # 3) Exploration des Testdatensatzes
    explore_single_data()

    # 4) Beispiel-Vorhersage
    try:
        example_input = [{
            "startYear": 2010,
            "isAdult": 0,
            "num_languages": 2,
            "has_originalTitle_aka": 1,
            "directors_hashed": 1234,
            "writers_hashed": 5678,
            "categories_hashed": 99999,
            "numVotes": 5000,
            "titleType_hashed": 42
        }]
        predictions = use_single_data(example_input)
        print("Beispiel-Vorhersage:", predictions)
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()