import dukpy
from os.path import abspath, dirname, join

class UPEmulator:
    # Names and millisecond intervals of interval loops
    _interval_loops = {
        'intervalLoop1': 1000,
        'intervalLoop2': 2500,
        'intervalLoop3': 10,
        'intervalLoop4': 100
    } 
    
    _obs_to_js = {
        'Paperclips': 'clips',
        'Available Funds': 'funds',
        'Unsold Inventory': 'unsoldClips',
        'Price per Clip': 'margin',
        'Public Demand': 'demand',
        'Marketing Level': 'marketingLvl',
        'Marketing Cost': 'adCost',
        'Manufacturing Clips per Second': 'clipRateTracker',
        'Wire Inches': 'wire',
        'Wire Cost': 'wireCost',
        'Autoclipper Cost': 'clipperCost',
        'Number of Autoclippers': 'clipmakerLevel'
    }
    
    _action_to_js = {
        'Make Paperclip': 'clipClick(1)',
        'Lower Price': 'lowerPrice()',
        'Raise Price': 'raisePrice()',
        'Expand Marketing': 'buyAds()',
        'Buy Wire': 'buyWire()',
        'Buy Autoclipper': 'makeClipper()'
    }
    
    def __init__(self, 
                 combat_filename=join(dirname(abspath(__file__)),"combat.js"),
                 globals_filename=join(dirname(abspath(__file__)),"globals.js"),
                 projects_filename=join(dirname(abspath(__file__)),"projects.js"),
                 main_filename=join(dirname(abspath(__file__)),"main.js")):

        # Source file list
        self._js_filenames = [combat_filename, globals_filename, projects_filename, main_filename]
        self._init()
    
    def _init(self):
        # Set up interpreter
        self._intp = dukpy.JSInterpreter()
        
        # Make initial source read
        for fname in self._js_filenames:            
            with open(fname, "r") as f:
                self._intp.evaljs(f.read())
    
    # "Public" members
    def reset(self):
        self._init()
    
    def quit(self):
        # Nothing needs to be done here
        pass
    
    def makeObservation(self, fields):
        obs = {}
        for field in fields:
            obs[field] = self._intp.evaljs(self._obs_to_js[field])
        return obs
        
    def takeAction(self, action_name):
        # @todo Check if action should actually be available
        self._intp.evaljs(self._action_to_js[action_name])
