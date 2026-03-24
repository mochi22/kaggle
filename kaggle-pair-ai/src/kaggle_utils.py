from kaggle.api.kaggle_api_extended import KaggleApi

def get_active_competitions(limit=20):
    api = KaggleApi()
    api.authenticate()
    competitions = api.competitions_list(sort_by='latestDeadline')
    return [{
        "ref": comp.ref,
        "title": comp.title,
        "deadline": comp.deadline
    } for comp in competitions[:limit]]

import requests
from bs4 import BeautifulSoup

def get_competition_description(ref):
    url = f"{ref}/overview"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve the competition page: {url}")
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # return all HTML in overview
    return soup

import json
import os
import requests
from bs4 import BeautifulSoup

def get_and_save_competition_description(ref):
    ref = ref.split("/")[-1]
    # 保存ディレクトリの作成
    os.makedirs('datas/competitions', exist_ok=True)
    
    url = f"https://www.kaggle.com/competitions/{ref}/overview"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve the competition page: {url}")
    
    soup = str(BeautifulSoup(response.text, 'html.parser'))
    print(soup)
    
    # データをJSON形式で保存
    file_path = f"datas/competitions/{ref}.html"
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(soup)
    return file_path  # 保存したファイルパスを返す

def get_and_save_competition_discussion(ref):
    ref = ref.split("/")[-1]
    # 保存ディレクトリの作成
    os.makedirs('datas/discussions', exist_ok=True)
    
    url = f"https://www.kaggle.com/competitions/{ref}/discuss"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve the competition page: {url}")
    
    soup = str(BeautifulSoup(response.text, 'html.parser'))
    print(soup)
    
    # データをJSON形式で保存
    file_path = f"datas/discussions/{ref}.html"
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(soup)
    return file_path  # 保存したファイルパスを返す

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def get_and_save_competition_html(ref, link, save_path='datas/competitions'):
    ref = ref.split("/")[-1]

    # Seleniumのオプション設定
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # ヘッドレスモード（ブラウザを表示せずに処理）
    
    # ChromeDriverのセットアップ
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    # URLにアクセス
    url = f"https://www.kaggle.com/competitions/{ref}/{link}"
    driver.get(url)
    
    # ページの完全な読み込みを待機（必要に応じて調整）
    # WebDriverWait(driver, 20).until(
    #     EC.presence_of_element_located((By.ID, "root"))
    # )
    driver.implicitly_wait(10)

    # ディスカッションへのリンクを取得（リンクは<a>タグに含まれていることが多い）
    # if link == "discussion":
    #     discussion_links = []
    #     discussion_elements = driver.find_elements(By.CLASS_NAME, "a[class*='sc-lgprfV eLusIR']")
        
    #     for element in discussion_elements:
    #         relative_url = element.get_attribute("href")  # 相対URL
    #         print("rel:", relative_url)
    #         if relative_url:  # 相対URLが存在する場合
    #             discussion_links.append(relative_url)
    #     print(discussion_links)
    # # class="sc-lgprfV eLusIR"のリンクをすべて取得
    links = driver.find_elements(By.CLASS_NAME, "sc-lgprfV.eLusIR")

    # リンクを表示
    for link in links:
        href = link.get_attribute("href")
        print(f"https://www.kaggle.com{href}")

    
    # ページのHTMLを取得
    page_source = driver.page_source
    
    # 保存ディレクトリの作成
    os.makedirs(save_path, exist_ok=True)
    
    # HTMLを保存
    file_path = f"{save_path}/{ref}.html"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(page_source)
    
    driver.quit()  # ブラウザを閉じる
    
    return file_path  # 保存したファイルパスを返す


"""
import sqlite3
from datetime import datetime

# SQLiteデータベース接続
def get_db_connection():
    conn = sqlite3.connect('src/database/kaggle_data.db')
    return conn

# データベーステーブル作成（必要な場合）
def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS competitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ref TEXT NOT NULL,
        datas TEXT NOT NULL,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

# コンペ情報を取得（必要に応じて）
def get_competition_from_db(ref):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM competitions WHERE ref = ?', (ref,))
    comp = cursor.fetchone()
    conn.close()
    return comp

def save_competition_to_db(ref, datas):
    conn = get_db_connection()  # SQLiteデータベースへの接続を取得
    create_tables()
    cursor = conn.cursor()  # SQL操作用のカーソルを作成
    cursor.execute('''
    INSERT INTO competitions (ref, datas)
    VALUES (?, ?)
    ''', (ref, datas))  # コンペ情報をINSERT文でデータベースに保存
    conn.commit()  # 変更をデータベースに反映
    conn.close()  # 接続を閉じる
"""
