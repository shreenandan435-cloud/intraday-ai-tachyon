import os
import sys
import requests

# Try to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")

print("--- TELEGRAM DIAGNOSTIC ---")
print(f"TELEGRAM_BOT_TOKEN present: {'YES (' + bot_token[:5] + '...)' if bot_token else 'NO (EMPTY)'}")
print(f"TELEGRAM_CHAT_ID present:   {'YES (' + str(chat_id) + ')' if chat_id else 'NO (EMPTY)'}")

if not bot_token or not chat_id:
    print("\n[FAIL] Environment variables are missing or empty.")
    print("If you have a .env file, ensure python-dotenv is installed or set them in PowerShell:")
    print('  $env:TELEGRAM_BOT_TOKEN="your_bot_token"')
    print('  $env:TELEGRAM_CHAT_ID="your_chat_id"')
    sys.exit(1)

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
payload = {
    "chat_id": chat_id,
    "text": "🔔 *Tachyon Diagnostic Test*: Synchronous link verified.",
    "parse_mode": "Markdown"
}

print("\nDispatching test message to Telegram servers...")
try:
    response = requests.post(url, json=payload, timeout=10.0)
    print(f"HTTP Status Code: {response.status_code}")
    print(f"Telegram API Response: {response.text}")
    
    if response.status_code == 200:
        print("\n[SUCCESS] Message successfully delivered to Telegram!")
    else:
        print("\n[FAIL] Telegram rejected the request. Check the response message above.")
except Exception as e:
    print(f"\n[ERROR] Connection failed: {e}")
