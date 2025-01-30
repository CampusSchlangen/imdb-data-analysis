# Meilenstein 2

## Ziel der Aufgabe
Das Ziel der Aufgabe ist es, mithilfe eines Modells die durchschnittliche Bewertung eines Films, einer Serie oder eines ähnlichen Werkes vorherzusagen. Dazu werden Daten von IMDb verwendet.

## Datenvorverarbeitung
Die Daten stammen aus einem Datenbank-Dump von IMDb und benötigen erwartungsgemäß wenig Vorverarbeitung. Dennoch wird sichergestellt, dass der Datensatz keine leeren Spalten oder Zeilen enthält – solche werden entfernt.

Im Zuge der Datenvorverarbeitung wird die Spalte `titleId` in der Datenbanktabelle `title_akas` in `tconst` umbenannt. Dies erleichtert das spätere Merging der einzelnen Datenbanktabellen zu einem umfassenden Datensatz.

## Datenexploration
Die Datenexploration wurde genutzt, um den Einfluss potenzieller Features auf das `AvgRating` eines Titels zu visualisieren.

Beispielsweise hat die Laufzeit keinen Einfluss auf die Bewertung, während die Art des Werkes eine Rolle spielt. So sind Filme tendenziell schlechter bewertet als andere Arten von Werken, während Serienepisoden tendenziell besser bewertet sind.

## Features
Für das Training wurden bislang folgende Features festgelegt:

- `titleType` (gehasht)
- `isAdult`
- `startYear`
- `category` (gehasht)
- `genres` (gehasht)
- `writer` (gehasht)
- `directors` (gehasht)

## Ergebnis
Je nach Modellart liegt der aktuelle R²-Score zwischen 0,3 und 0,4. Dabei gilt: Je näher der Wert an 1 liegt, desto besser trifft das Modell mit seinen Vorhersagen das tatsächliche Rating.

## Ausblick
Folgende Verbesserungen sind für die Zukunft geplant:

- **StratifiedShuffleSplit**, um den Datensatz zu verkleinern, ohne das Risiko einzugehen, dass manche Features in einem unrealistischen Verhältnis zueinander stehen (z. B. `titleType`).
- **OrdinalEncoder** anstelle von Hash-Werten.
- **Permutation Feature Importance**, um zu bestimmen, welche Features welchen Einfluss auf das Endmodell haben – eine genauere Methode als die reine visuelle Datenexploration.
