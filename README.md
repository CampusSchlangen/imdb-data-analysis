# IMDb Datenverarbeitung

## Projektübersicht
In diesem Projekt wird eine komplette Pipeline zur Verarbeitung, Modellierung und Analyse von IMDb-Daten bereitgestellt.  
Ziel ist es, **IMDb-TSV-Dateien** einzulesen, zu bereinigen, Features zu erzeugen und anschließend **Machine-Learning-Modelle** zu trainieren, um die durchschnittliche Bewertung (`averageRating`) von Filmen und Serien vorherzusagen.

---

## Verzeichnisstruktur

```plaintext
.
├── data/                      # Originale IMDb TSV-Datensätze
├── cleaned_data_single/       # Bereinigte Datensätze (Einzeln)
├── master_features/           # Master-Dataset mit aggregierten Features
├── master_train_data/         # Trainingsdatensatz (Master)
├── master_test_data/          # Testdatensatz (Master)
├── master_models/             # Trainierte Modelle (Master)
├── master_exploration/        # Visualisierungen & Diagramme
├── main.py                    # Hauptskript mit allen Funktionen
└── README.md                  # Diese Datei
```

## Voraussetzungen
### 1. Python-Version

- Python 3.8 oder höher wird empfohlen

### 2. Benötigte Bibliotheken

Installiere die folgenden Bibliotheken, um alle Features nutzen zu können:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```
```bash
pip install xgboost lightgbm
```

## Funktionen und Skriptaufbau

Alle wichtigen Schritte sind in main.py enthalten. Dort findest Du folgende Hauptfunktionen:

### prepare_single()
Liest mehrere IMDb-TSV-Dateien aus dem Verzeichnis data/.
Bereinigt die Daten (Entfernung leerer Zeilen/Spalten, Konvertierung von Typen).
Merged die Datensätze (u. a. title_basics, title_ratings, title_crew und title_principals).
Erstellt einen Master-Datensatz und führt einen Train-Test-Split durch.
Speichert die Ergebnisse in master_features/, master_train_data/ und master_test_data/.

### train_single_data()
Lädt den Trainingsdatensatz (master_train_data/master_train.csv).
Trennt Features und Ziel (hier averageRating).
Testet verschiedene ML-Modelle (Ridge, Random Forest, SVR, HistGradientBoosting usw.) über GridSearchCV.
Speichert das beste Modell im Verzeichnis master_models/.

### explore_single_data()
Nimmt den Testdatensatz (master_test_data/master_test.csv).
Erstellt Histogramme, Scatterplots und Boxplots zu averageRating und den restlichen Features.
Speichert die Diagramme in master_exploration/.

### use_single_data(input_data)
Lädt das bereits trainierte Modell aus master_models/master_model.joblib.
Führt Vorhersagen auf neuen Daten durch.
Gibt die geschätzten Bewertungen (averageRating) zurück.

### main()
Orchestriert den gesamten Ablauf:
prepare_single()
train_single_data()
explore_single_data()
Beispiels-Vorhersage mit use_single_data()  

Nutzung

IMDb-TSV-Dateien vorbereiten
Lege die benötigten IMDb-Dateien (z. B. title.basics.tsv, title.ratings.tsv, title.crew.tsv, title.principals.tsv) in den Ordner ./data/.

Pipeline ausführen
Starte das Skript im Termin 
```bash
python main.py
```

## Dadurch werden sämtliche Schritte durchgeführt:

Datenaufbereitung
Feature-Erzeugung & Zusammenführung
Training
Testdatensatz-Exploration
Beispiel-Vorhersage
