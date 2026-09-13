import logging
import requests
import threading

class TelegramAlerter:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logging.getLogger("Tachyon.Telegram")

    def send(self, message: str):
        if not self.bot_token or not self.chat_id:
            return
            
        def _dispatch():
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
            try:
                requests.post(url, json=payload, timeout=2.0)
            except Exception as e:
                self.logger.warning(f"Telegram dispatch failed: {e}")
                
        # Run on background thread to prevent blocking the quant loop
        threading.Thread(target=_dispatch, daemon=True).start()
