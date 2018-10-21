from pynput.keyboard import Key, Controller

def type_key(key):
    keyboard = Controller()
    keyboard.press(key)
    keyboard.release(key)