from py_mini_racer import py_mini_racer
from os.path import abspath, dirname, join
from collections import OrderedDict
from upb.game.UPGameHandler import UP_PROJECT_IDS

class UPEmulator(object):
    # Names and centisecond intervals of interval loops that run over the full game
    _interval_loops_cs = {
        'intervalLoop1()': 100,
        'intervalLoop2()': 250,
        'intervalLoop3()': 1,
        'intervalLoop4()': 10
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
        'Creativity': 'creativity',
        'Investment Bankroll': 'bankroll',
        'Stocks': 'secTotal',
        'Investment Engine Upgrade Cost': 'investUpgradeCost',
        'Number of Photonic Chips': 'nextQchip',
        'Photonic Chip 0 Level': 'qChips[0].value',
        'Riskiness': 'riskiness',
        'MegaClipper Cost': 'megaClipperCost',
        'Number of MegaClippers': 'megaClipperLevel',
        'Yomi': 'yomi',
        'Tournament Cost': 'tourneyCost'
    }
    for pname, pid in UP_PROJECT_IDS.items():
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
        'Set Investment Low': 'if (investmentEngineFlag == 1) {riskiness = 7;}',
        'Set Investment Medium': 'if (investmentEngineFlag == 1) {riskiness = 5;}',
        'Set Investment High': 'if (investmentEngineFlag == 1) {riskiness = 1;}',
        'Withdraw': 'if (investmentEngineFlag == 1) {investWithdraw();}',
        'Deposit': 'if (investmentEngineFlag == 1) {investDeposit();}',
        'Upgrade Investment Engine': 'if (investmentEngineFlag == 1 && yomi>=investUpgradeCost) {investUpgrade();}',
        'Quantum Compute': 'if (qFlag == 1) {qComp();}',
        'Buy MegaClipper': 'if (funds>=megaClipperCost && megaClipperFlag == 1) {makeMegaClipper();}',
    }
    for pname, pid in UP_PROJECT_IDS.items():
        _action_to_js['Activate '+pname] = 'if (activeProjects.indexOf(project{0}) >= 0 && project{0}.cost() && !project{0}.flag) {{project{0}.effect();}}'.format(pid)
    
    def __init__(self, 
                 combat_filename=join(dirname(abspath(__file__)),"combat.js"),
                 globals_filename=join(dirname(abspath(__file__)),"globals.js"),
                 projects_filename=join(dirname(abspath(__file__)),"projects.js"),
                 main_filename=join(dirname(abspath(__file__)),"main_pre_drones.js")):

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
        self._loop_counters = {}
        for loop_name in self._interval_loops_cs.keys():
            self._loop_counters[loop_name] = 0
        
        # Other init
        self._tourney_running = False
        
        # Allow primary main loops to resolve at least once
        self.advanceTime(0.5)
    
    def quit(self):
        # Nothing needs to be done here
        pass
    
    def makeObservation(self, fields):
        obs = OrderedDict()
        
        # Package as a javascript array
        fields_js = "["
        for i in range(len(fields)-1):
            fields_js += self._obs_to_js[fields[i]]+","
        fields_js += self._obs_to_js[fields[-1]]
        fields_js += "]"
        
        # Fetch array of values
        vals = self._intp.eval(fields_js)
        
        # Populate our array        
        for i in range(len(fields)):
            field = fields[i]
            val = vals[i]
            
            # Any additional scaling that's made before being displayed to webpage
            if field == 'Public Demand':
                val *= 10.0
                
            obs[field] = val
        return obs
    
    def runNewTournament(self):
        tourney_possible = self._intp.eval("strategyEngineFlag == 1 && operations>=tourneyCost && tourneyInProg == 0")
        if tourney_possible:
            # Check we haven't messed something up
            if self._tourney_running:
                raise Exception("Python and JS tournament logic have become incompatible somehow.")             
            
            # Create new tournament
            self._intp.eval("newTourney();")
            
            # Make a pick
            # For the moment, just pick 0, but can be made random or selected in future
            self._intp.eval("pick=0;")
            
            # The tournament takes 100 ms per round. 
            # See the setTimeout arguments in "function round(roundNum)".
            tourney_run_time_cs = 10*self._intp.eval("rounds") 
            self._tourney_end_time_cs = self._time_cs+tourney_run_time_cs
            self._tourney_running = True
    
    def takeAction(self, action_name):
        if action_name == "Run New Tournament":
            self.runNewTournament()
        else:
            self._intp.eval(self._action_to_js[action_name])
        
    def advanceTime(self, dt_s):
        dt_cs = max(int(100.0*dt_s),1)
        time_cs_next = self._time_cs + dt_cs
        for loop_name, loop_interval in self._interval_loops_cs.items():
            # Determine how many iterations of the loop to perform
            loop_time_cs = self._loop_counters[loop_name]*self._interval_loops_cs[loop_name]
            time_to_advance_cs = time_cs_next - loop_time_cs
            iters_to_advance = int(time_to_advance_cs/self._interval_loops_cs[loop_name])
            
            # Perform the iterations
            if iters_to_advance > 0:
                loop_js = "for (var i = 0; i < {}; i++) {{ {} ;}}".format(iters_to_advance, loop_name)
                self._intp.eval(loop_js)
            
            # Update the execution count for the loop
            self._loop_counters[loop_name] += iters_to_advance
        self._time_cs = time_cs_next
        
        # Check for tournament completion
        if self._tourney_running:
            # Complete the tournament by running all its rounds instantly
            if self._time_cs > self._tourney_end_time_cs:
                self._intp.eval("instantRunTourney();")
                self._tourney_running = False
    
    def getStateAsString(self):
        stateString = self._intp.eval("getSaveAsString();")
        return stateString
    
    def saveState(self, filename):
        # Complete any tournaments before saving state
        tourney_time_remaining_cs = self._tourney_end_time_cs - self._time_cs
        if self._tourney_running:
            self.advanceTime((tourney_time_remaining_cs+2.0)/100.0)
        assert not self._tourney_running
        
        # Save
        with open(filename, 'w') as f:
            f.write(self.getStateAsString())
            
    def loadStateFromString(self, stateString):
        self._intp.eval("loadStateFromString({});".format(stateString))
        
    def loadState(self, filename):
         with open(filename, 'r') as f:
            self.loadStateFromString(f.read())
