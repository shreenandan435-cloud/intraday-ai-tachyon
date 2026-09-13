from telegram_notifier import TelegramNotifier
import os

notifier = TelegramNotifier()
print("[+] Dispatching test notification to Telegram...")
success = notifier.send(
    "⚡ *TACHYON TEST ALERT*\n"
    "─────────────────────\n"
    "• *System:* Geometric Breakout Engine\n"
    "• *Broker Link:* Angel One SmartAPI\n"
    "• *Pipeline Status:* Operational & Alert Verified.\n"
    "─────────────────────\n"
    "If you received this message, mobile alerts are active."
)

if success:
    print("[SUCCESS] Telegram alert confirmed on your mobile device.")
else:
    print("[!] Verification failed. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
