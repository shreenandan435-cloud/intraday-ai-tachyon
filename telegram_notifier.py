import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else None
        
        if not self.bot_token or not self.chat_id:
            logging.warning("[!] Telegram credentials missing in .env. Alerts will print to console only.")

    def send(self, message: str, silent: bool = False):
        """Sends a Markdown-formatted message to Telegram."""
        if not self.api_url or not self.chat_id:
            print(f"[TELEGRAM SIMULATED]:\n{message}")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_notification": silent
        }
        
        try:
            resp = requests.post(self.api_url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logging.error(f"[!] Telegram alert failed: {str(e)}")
            return False

    def notify_breakout_entry(self, symbol: str, entry_price: float, sl: float, target: float, qty: int, rvol: float):
        msg = (
            f"🚀 *BREAKOUT ENTRY EXECUTED*\n"
            f"─────────────────────\n"
            f"• *Symbol:* `{symbol}`\n"
            f"• *Entry Limit:* `₹{entry_price:.2f}`\n"
            f"• *Stop Loss:* `₹{sl:.2f}`\n"
            f"• *Target (1:2):* `₹{target:.2f}`\n"
            f"• *Quantity:* `{qty}` shares\n"
            f"• *RVOL:* `{rvol:.2f}x`\n"
            f"─────────────────────\n"
            f"⚡ _Status: Marketable Limit Order Dispatched_"
        )
        return self.send(msg)

    def notify_exit(self, symbol: str, exit_price: float, pnl_rupees: float, reason: str):
        outcome_icon = "🟢" if pnl_rupees >= 0 else "🔴"
        msg = (
            f"{outcome_icon} *POSITION CLOSED*\n"
            f"─────────────────────\n"
            f"• *Symbol:* `{symbol}`\n"
            f"• *Exit Price:* `₹{exit_price:.2f}`\n"
            f"• *Net P&L:* `₹{pnl_rupees:+.2f}`\n"
            f"• *Reason:* `{reason}`\n"
            f"─────────────────────"
        )
        return self.send(msg)

    def notify_heartbeat(self, active_symbols: list, active_trades: int):
        msg = (
            f"💓 *TACHYON HEARTBEAT*\n"
            f"• *Engine:* Autonomous Geometric Breakout\n"
            f"• *Tracked Basket:* `{', '.join(active_symbols)}`\n"
            f"• *Open Positions:* `{active_trades}`\n"
            f"• *Status:* Online & Scanning Mode-3 WebSocket"
        )
        return self.send(msg, silent=True)
