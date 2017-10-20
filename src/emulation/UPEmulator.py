import dukpy

class UPEmulator:
    def __init__(self, 
                 combat_filename="combat.js",
                 globals_filename="globals.js",
                 projects_filename="projects.js"):
        
        # Set up interpreter
        self.intp = dukpy.JSInterpreter()
        
        # Make initial source read
        combat_file = open(combat_filename, "r")
        globals_file = open(globals_filename, "r")
        projects_file = open(projects_filename, "r")
        self.intp.evaljs(combat_file.read())
        self.intp.evaljs(globals_file.read())
        self.intp.evaljs(projects_file.read())
        
    def _getVal(self, name):
        return self.intp.evaljs(name)
        
emu = UPEmulator()
print(emu._getVal("wireCost"))
