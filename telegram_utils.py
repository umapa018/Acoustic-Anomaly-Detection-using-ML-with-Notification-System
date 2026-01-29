import requests
import os
from dotenv import load_dotenv
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        res = requests.post(url, data=payload)
        if res.status_code == 200:
            print("Telegram message sent sucessfull! ")
        else:
            print("Failed to send Telegram message ", res.text)
    except Exception as e:
        print("Error:", e)
