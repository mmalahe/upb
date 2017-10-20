import dukpy
from os.path import abspath, dirname, join

class UPEmulator:
    def __init__(self, 
                 combat_filename=join(dirname(abspath(__file__)),"combat.js"),
                 globals_filename=join(dirname(abspath(__file__)),"globals.js"),
                 projects_filename=join(dirname(abspath(__file__)),"projects.js"),
                 main_filename=join(dirname(abspath(__file__)),"main.js")):
        
        # Set up interpreter
        self.intp = dukpy.JSInterpreter()
        
        # Make initial source read
        for fname in [combat_filename, globals_filename, projects_filename, main_filename]:            
            with open(fname, "r") as f:
                self.intp.evaljs(f.read())
                
        # Time intervals for interval loops
        self.interval_loops = {
            'intervalLoop1': 1000,
            'intervalLoop2': 2500,
            'intervalLoop3': 10,
            'intervalLoop4': 100
        }
        
    def _getVal(self, name):
        return self.intp.evaljs(name)
        
emu = UPEmulator()
print(emu._getVal("wireCost"))
