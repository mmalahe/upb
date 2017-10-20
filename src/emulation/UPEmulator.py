import dukpy

class UPEmulator:
    def __init__(self, 
                 combat_filename="combat.js",
                 globals_filename="globals.js",
                 projects_filename="projects.js",
                 main_filename="main.js"):
        
        # Set up interpreter
        self.intp = dukpy.JSInterpreter()
        
        # Make initial source read
        for fname in [combat_filename, globals_filename, projects_filename, main_filename]:            
            with open(fname, "r") as f:
                self.intp.evaljs(f.read())
                
        # Time intervals for interval loops
        intervalLoops = {
            'intervalLoop1': 1000,
            'intervalLoop2': 2500,
            
        }
        
    def _getVal(self, name):
        return self.intp.evaljs(name)
        
emu = UPEmulator()
print(emu._getVal("wireCost"))
