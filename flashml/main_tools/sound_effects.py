import os
import sys


def bell():
    """
    Plays a system notification sound (beep) to indicate completion.
    """
    import platform

    system_platform = platform.system()

    try:
        if system_platform == "Windows":
            # Use the winsound module on Windows
            import winsound

            # Play the default asterisk sound, common for notifications
            # Other options: winsound.MB_OK, winsound.MB_ICONEXCLAMATION, etc.
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
            # print("Process finished (Windows sound played).")

        elif system_platform == "Darwin":  # macOS
            # On macOS, printing the BEL character often works in the terminal
            # Alternatively, use afplay for a specific sound file
            print("\a")
            sys.stdout.flush()  # Ensure the character is printed immediately
            # print("Process finished (macOS alert attempted).")
            # Uncomment below to play a specific system sound (if \a doesn't work)
            # os.system('afplay /System/Library/Sounds/Sosumi.aiff')

        elif system_platform == "Linux":
            # On Linux, printing the BEL character to stdout usually works
            # if the terminal supports it and system sounds are configured.
            print("\a")
            sys.stdout.flush()  # Ensure the character is printed immediately
            # print("Process finished (Linux alert attempted).")
            # Alternative for systems using PulseAudio (might need paplay installed)
            # os.system('paplay /usr/share/sounds/freedesktop/stereo/dialog-information.oga')
            # Alternative for systems using ALSA (might need aplay installed)
            # os.system('aplay /usr/share/sounds/alsa/Front_Center.wav')

        else:
            # Fallback for other or unknown systems: try the BEL character
            print("\a")
            sys.stdout.flush()
            print(f"Process finished (Generic alert attempted on {system_platform}).")

    except Exception as e:
        # print(f"Could not play sound due to an error: {e}")
        # Still print the BEL character as a last resort
        print("\a")
        sys.stdout.flush()
        # print("Process finished (sound failed, fallback alert attempted).")


def play_waiting_sound():
    assert "Not working for now"
    from playsound import playsound

    base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes from tools -> flashml
    file_path = os.path.join(base_dir, "assets", "elevator_waiting_sound.mp3")
    file_path = os.path.normpath(file_path)  # Normalizes slashes for Windows
    try:
        playsound(file_path)
        print(f"Playing sound: {file_path}")
    except Exception as e:
        print(f"Error playing sound: {e}")
