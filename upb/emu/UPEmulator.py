from py_mini_racer import py_mini_racer
from os.path import abspath, dirname, join
from collections import OrderedDict

class UPEmulator(object):
    # Names and centisecond intervals of interval loops
    _interval_loops_cs = {
        'intervalLoop1()': 100,
        'intervalLoop2()': 250,
        'intervalLoop3()': 1,
        'intervalLoop4()': 10
    } 
    
    _project_ids = {
        'Improved AutoClippers': '1',
        'Beg for More Wire': '2',
        'Creativity': '3',
        'Even Better AutoClippers': '4',
        'Optimized AutoClippers': '5',
        'Limerick': '6',
        'Improved Wire Extrusion': '7',
        'Optimized Wire Extrusion': '8',
        'Microlattice Shapecasting': '9',
        'New Slogan': '11',
        'Catchy Jingle': '12',
        'Lexical Processing': '13',
        'Combinatory Harmonics': '14',
        'The Hadwiger Problem': '15',
        'The Toth Sausage Conjecture': '17',
        'Hadwiger Clip Diagrams': '16',
        'Donkey Space': '19',
        'Algorithmic Trading': '21',
        'WireBuyer': '26',
        'Hypno Harmonics': '34',
        'RevTracker': '42'
    }
    
    # Observations
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
        'Number of Autoclippers': 'clipmakerLevel',
        'Trust': 'trust',
        'Next Trust': 'nextTrust',
        'Processors': 'processors',
        'Memory': 'memory',
        'Operations': 'operations',
        'Creativity': 'creativity'
    }
    for pname, pid in _project_ids.items():
        _obs_to_js[pname+' Activated'] = 'project{}.flag'.format(pid)
    
    # Actions
    _action_to_js = {
        'Make Paperclip': 'if (wire>0) {clipClick(1);}',
        'Lower Price': 'if (margin>.01) {lowerPrice();}',
        'Raise Price': 'raisePrice();',
        'Expand Marketing': 'if (funds>=adCost) {buyAds();}',
        'Buy Wire': 'if (funds>=wireCost) {buyWire();}',
        'Buy Autoclipper': 'if (funds>=clipperCost) {makeClipper();}',
        'Add Processor': 'if (trust>processors+memory || swarmGifts > 0) {addProc();}',
        'Add Memory': 'if (trust>processors+memory || swarmGifts > 0) {addMem();}',
    }
    for pname, pid in _project_ids.items():
        _action_to_js['Activate '+pname] = 'if (activeProjects.indexOf(project{0}) >= 0 && project{0}.cost() && !project{0}.flag) {{project{0}.effect();}}'.format(pid)
    
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
        self._intp = py_mini_racer.MiniRacer()
        self._time_cs = 0
        
        # Make initial source read
        for fname in self._js_filenames:            
            with open(fname, "r") as f:
                self._intp.eval(f.read())
    
    # "Public" members
    def reset(self):
        self._init()
        # Allow all main loops to resolve at least once
        self.advanceTime(0.5)
    
    def quit(self):
        # Nothing needs to be done here
        pass
    
    def makeObservation(self, fields):
        obs = OrderedDict()
        for field in fields:
            val = self._intp.eval(self._obs_to_js[field])
            
            # Any additional scaling that's made before being displayed to webpage
            if field == 'Public Demand':
                val *= 10.0
                
            obs[field] = val
        return obs
        
    def takeAction(self, action_name):
        self._intp.eval(self._action_to_js[action_name])
        
    def advanceTime(self, dt_s):
        dt_cs = max(int(100.0*dt_s),1)
        for i in range(dt_cs):
            self._time_cs += 1
            for loop_name, loop_interval in self._interval_loops_cs.items():
                if self._time_cs % loop_interval == 0:
                    self._intp.eval(loop_name)
