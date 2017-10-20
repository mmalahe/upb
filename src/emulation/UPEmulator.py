import dukpy

globals_js = open("globals.js", "r")

class UPEmulator:
    def __init__(self):
        pass
        
interpreter = dukpy.JSInterpreter()
interpreter.evaljs(globals_js.read())
wireCost = interpreter.evaljs("wireCost")
