



def send_telegram_notification(message: str, chat_id: str|int = "7759637947", bot_token: str = "8038730906:AAHXrl-PRXEM4XR3fFzBXyWAOT-BMQqYea8", async_mode: bool = False):
    from telegram import Bot
    # Synchronous sending (default, safe for atexit)
    if not async_mode:
        import requests
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Error sending message: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Async sending (only when explicitly requested)
    async def send_message():
        try:
            bot = Bot(token=bot_token)
            async with bot:
                await bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
        except Exception as e:
            print(f"Error sending message: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, send_message())
                future.result()
        except RuntimeError:
            asyncio.run(send_message())
    except Exception as e:
        print(f"Error in asyncio: {e}")
        import traceback
        traceback.print_exc()