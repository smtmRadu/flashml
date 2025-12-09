


def send_telegram_notification(message: str, chat_id: str = "7759637947", bot_token: str = "8038730906:AAHXrl-PRXEM4XR3fFzBXyWAOT-BMQqYea8"):
    import asyncio
    from telegram import Bot
    
    async def send_message():
        try:
            bot = Bot(token=bot_token)
            async with bot:
                await bot.send_message(chat_id=chat_id, text=message)
            # print("Message sent successfully!")
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

