import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import time
from imdb import IMDb
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random


# --- Scraping der Anime-Daten von IMDb mit Selenium für 'Mehr laden' ---
def get_all_titles_from_imdb():
    base_url = "https://www.imdb.com/search/title/?genres=animation&title_type=movie,tv_series,tv_special,ova"
    titles_set = set()  # Set zur Vermeidung von Duplikaten

    # User-Agent Rotation
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.88 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:110.0) Gecko/20100101 Firefox/110.0'
    ]

    # Selenium WebDriver initialisieren
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Headless-Modus
    options.add_argument(f'user-agent={random.choice(user_agents)}')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--lang=de-DE')  # Seite auf Deutsch anfordern
    driver = webdriver.Chrome(options=options)
    driver.get(base_url)

    while True:
        try:
            load_more_buttons = driver.find_elements(By.XPATH, "//button//span[contains(text(), '50 mehr')]")
            if load_more_buttons:
                driver.execute_script("arguments[0].scrollIntoView(true);", load_more_buttons[0])
                time.sleep(1)  # Warten auf Rendern nach dem Scrollen
                load_more_buttons[0].click()
                print("Button '50 mehr' gefunden und geklickt.")
                time.sleep(2)
            else:
                print("Button '50 mehr' nicht gefunden.")
                break
        except Exception as e:
            print(f"Fehler beim Klicken auf 'Mehr laden': {e}")
            break

    # Nach vollständigem Laden die Titel extrahieren
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    titles = soup.find_all('div', class_='sc-300a8231-0 gTnHyA')

    for item in titles:
        title_tag = item.find('h3', class_='ipc-title__text')
        if title_tag:
            titles_set.add(title_tag.text)  # Fügt Titel zu Set hinzu
            print(f"Titel gefunden: {title_tag.text}.")
        time.sleep(0.1)

    print(f"{len(titles_set)} einzigartige Titel auf der Seite gespeichert.")
    driver.quit()
    return list(titles_set)  # Konvertiere Set in Liste


# --- Methode für den HTML-Dump mit Selenium ---
def save_imdb_page_to_txt(url, output_file='imdb_page_dump.txt'):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.88 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:110.0) Gecko/20100101 Firefox/110.0'
    ]

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument(f'user-agent={random.choice(user_agents)}')
    options.add_argument('--lang=de-DE')
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
        print(f"Seite gespeichert in {output_file}.")

    driver.quit()


# --- IMDbPY zur Detailabfrage nutzen ---
def get_anime_details(titles_list):
    ia = IMDb()
    anime_list = []

    for idx, title in enumerate(titles_list):
        results = ia.search_movie(title)
        if results:
            movie = results[0]
            ia.update(movie)
            anime_list.append({
                'Titel': movie.get('title'),
                'Jahr': movie.get('year'),
                'Typ': movie.get('kind'),
                'Bewertung': movie.get('rating', 'N/A'),
                'Genres': ', '.join(movie.get('genres', [])),
                'Regisseur': ', '.join([person['name'] for person in movie.get('directors', [])])
            })
            print(f"{idx + 1}/{len(titles_list)} verarbeitet: {movie.get('title')}")
        time.sleep(0.5)

    return anime_list


# --- Daten in SQLite speichern mit Fehlerbehandlung ---
def save_to_sqlite(anime_list, db_name='anime_data.db'):
    if not anime_list:
        print("Keine Daten zum Speichern vorhanden. Vorgang abgebrochen.")
        return

    try:
        conn = sqlite3.connect(db_name)
        df = pd.DataFrame(anime_list)

        df.to_sql('anime', conn, if_exists='replace', index=False)
        conn.close()
        print(f"Daten in {db_name} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern in SQLite: {e}")


# --- Ausführen des Scraping-Prozesses und Speichern ---
if __name__ == "__main__":
    save_imdb_page_to_txt(
        "https://www.imdb.com/search/title/?genres=animation&title_type=movie,tv_series,tv_special,ova")
    titles = get_all_titles_from_imdb()
    anime_details = get_anime_details(titles)
    save_to_sqlite(anime_details)

    df = load_from_sqlite()
    print(df.head())
