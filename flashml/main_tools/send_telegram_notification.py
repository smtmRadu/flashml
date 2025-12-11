


def send_telegram_notification(message: str, chat_id: str, bot_token: str):
    import asyncio
    from telegram import Bot
    from telegram.constants import ParseMode
    
    async def send_message():
        try:
            bot = Bot(token=bot_token)
            async with bot:
                await bot.send_message(
                    chat_id=chat_id, 
                    text=message,
                    parse_mode=ParseMode.HTML  # Enable HTML parsing
                )
        except Exception as e:
            print(f"Error sending message: {e}")
            import traceback
            traceback.print_exc()
    
    try:
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
