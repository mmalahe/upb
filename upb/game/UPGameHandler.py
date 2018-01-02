from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
#~ from selenium.webdriver.support.select import Select
import selenium.common.exceptions
import selenium.webdriver.chrome as chrome
import lxml.html
import http
from collections import OrderedDict
import os

# Global constants
LOCAL_GAME_URL_TRAIN = "file://"+os.path.join(os.path.dirname(os.path.abspath(__file__)),"index2_train.html")
LOCAL_GAME_URL_STANDARD = "file://"+os.path.join(os.path.dirname(os.path.abspath(__file__)),"index2.html")
UP_PROJECT_IDS = {
    'Improved AutoClippers': '1',
    'Beg for More Wire': '2',
    'Creativity': '3',
    'Even Better AutoClippers': '4',
    'Optimized AutoClippers': '5',
    'Limerick': '6',
    'Improved Wire Extrusion': '7',
    'Optimized Wire Extrusion': '8',
    'Microlattice Shapecasting': '9',
    'Spectral Froth Annealment': '10',
    'Quantum Foam Annealment': '10b',
    'New Slogan': '11',
    'Catchy Jingle': '12',
    'Lexical Processing': '13',
    'Combinatory Harmonics': '14',
    'The Hadwiger Problem': '15',
    'Hadwiger Clip Diagrams': '16',
    'The Toth Sausage Conjecture': '17',
    'Toth Tubule Enfolding': '18',
    'Donkey Space': '19',
    'Strategic Modeling': '20',
    'Algorithmic Trading': '21',
    'MegaClippers': '22',
    'Improved MegaClippers': '23',
    'Even Better MegaClippers': '24',
    'Optimized MegaClippers': '25',
    'WireBuyer': '26',
    'Hypno Harmonics': '34',
    'RevTracker': '42',
    'Quantum Computing': '50',
    'Photonic Chip': '51',
    'New Strategy: A100': '60'
}

class UPGameState(object):
    _scalar_values = {}
    _scalar_values_finders = {
        'Paperclips': ['id','clips'],
        'Available Funds': ['id','funds'],
        'Unsold Inventory': ['id','unsoldClips'],
        'Price per Clip': ['id','margin'],
        'Public Demand': ['id','demand'],
        'Marketing Level': ['id','marketingLvl'],
        'Marketing Cost': ['id','adCost'],
        'Manufacturing Clips per Second': ['id','clipmakerRate'],
        'Wire Inches': ['id','wire'],
        'Wire Cost': ['id','wireCost'],
        'Autoclipper Cost': ['id','clipperCost'],
        'Number of Autoclippers': ['id','clipmakerLevel2'],
        'Processors': ['id','processors'],
        'Memory': ['id','memory'],
        'Trust': ['id','trust'],
        'Next Trust': ['id','nextTrust'],
        'Operations': ['id','operations'],
        'Creativity': ['id','creativity'],
        'Investment Bankroll': ['id', 'investmentBankroll'],
        'Stocks': ['id', 'secValue'],        
        #~ 'Riskiness', # Complicated, so implemented in __init__
        #~ 'Number of Photonic Chips',# Complicated, so implemented in __init__
        #~ 'Photonic Chip 0 Level', # Complicated, so implemented in __init__
        'Latest QOps': ['id', 'qCompDisplay'],
        'MegaClipper Cost': ['id', 'megaClipperCost'],
        'Number of MegaClippers': ['id', 'megaClipperLevel'],
        'Investment Engine Level': ['id', 'investmentLevel'],
        'Investment Engine Upgrade Cost': ['id', 'investUpgradeCost'],
        'Yomi': ['id', 'yomiDisplay'],
        'Tournament Cost': ['id', 'newTourneyCost']
    }
    _proj_avail_finders = {pname+' Available': ['id', 'projectButton'+UP_PROJECT_IDS[pname]] for pname in UP_PROJECT_IDS.keys()}
    
    def __init__(self, driver):        
        # The soup
        html_source = driver.find_element_by_xpath("//*").get_attribute("outerHTML")        
        html_parser = lxml.html.document_fromstring(html_source)
        
        # All kinds of values combined
        self._all_values = {}
        
        # Scalar values in the main text of elements
        for field, finder in self._scalar_values_finders.items():
            # Find value         
            if finder[0] == 'id':
                value = html_parser.get_element_by_id(finder[1]).text_content()
            else:
                raise NotImplementedError("Don't know how to find things by "+finder[0])
            
            # Convert value
            if value == None:
                self._scalar_values[field] = None
            else:
                multiplier = 1.0
                
                try:
                    self._scalar_values[field] = float(value)
                except:                    
                    # Cases where html spaces have been used
                    value = value.replace("&nbsp;","")
                    
                    # Assume commas are used to separate thousands, so simply remove them
                    value = value.replace(",","")
                    value = value.replace(u'\xa0',"")
                    
                    # Handle known cases where they're used to represent decimals
                    if field == 'Autoclipper Cost' or field == 'MegaClipper Cost':
                        multiplier /= 100.0
                        
                    # Case where no value is present yet
                    all_blank = True
                    for char in value:
                        if char != " " and char != "\n" and char != "\r":
                            all_blank = False
                    if all_blank:
                        value = "0"
                
                try:
                    self._scalar_values[field] = multiplier*float(value)
                except:
                    print(value)
        
        # More involved scalar values #
        ###############################
        
        # Photonic chip levels
        n_chips = 0
        for chip_number in range(1):        
            field = "Photonic Chip {} Level".format(chip_number)
            element_id = 'qChip{}'.format(chip_number)
            element = driver.find_element_by_id(element_id)
            opacity = float(element.value_of_css_property("opacity"))
            if opacity == 1: # Opacity is 1 when chip is inactive
                value = 0
            else:
                n_chips += 1
                value = opacity
            self._scalar_values[field] = value
        self._scalar_values['Number of Photonic Chips'] = n_chips
            
        # Investment riskiness
        invest_selector = html_parser.get_element_by_id('investStrat')
        riskiness = None
        for option in invest_selector:
            if 'selected' in option.attrib.keys():
                if option.attrib['selected'] == 'selected':
                    option_name = option.attrib['value']
                    if option_name == 'low':
                        riskiness = 7
                    elif option_name == 'med':
                        riskiness = 5
                    elif option_name == 'hi':
                        riskiness = 1
                    else:
                        raise Exception("Don't know what to do with \"{}\".".format(option_name))
        if riskiness == None:
            raise Exception("Should have a riskiness value.")
        self._scalar_values['Riskiness'] = riskiness        
        
        # Add scalar values to combined values
        self._all_values.update(self._scalar_values)
        
        # Project availability
        self._proj_avail = {}
        for field, finder in self._proj_avail_finders.items():      
            if finder[0] == 'id':
                value = html_parser.get_element_by_id(finder[1], None)
            else:
                raise NotImplementedError("Don't know how to find project availability by "+finder[0])                
            if value == None:
                self._proj_avail[field] = False
            else:
                self._proj_avail[field] = True
        self._all_values.update(self._proj_avail)
        
    def get(self, field):
        return self._all_values[field]
    
    def __str__(self):
        return self._scalar_values.__str__()        

class UPGameHandler(object):
    def __init__(self, 
                 url, 
                 webdriver_name='Chrome', 
                 webdriver_path=None, 
                 verbose=False, 
                 headless=False):
                     
        # Class constants
        self._all_buttons = {
            'Make Paperclip': 'btnMakePaperclip',
            'Lower Price': 'btnLowerPrice',
            'Raise Price': 'btnRaisePrice',
            'Expand Marketing': 'btnExpandMarketing',
            'Buy Wire': 'btnBuyWire',
            'Buy Autoclipper': 'btnMakeClipper',
            'Add Processor': 'btnAddProc',
            'Add Memory': 'btnAddMem'
        }
        self._project_buttons = {"Activate "+pname:"projectButton{}".format(pid) for pname, pid in UP_PROJECT_IDS.items()}
        self._all_buttons.update(self._project_buttons)
        self._all_actions = list(self._all_buttons.keys())
        
        # Action availability
        self._avail_observations = [
            'Paperclips',
            'Available Funds',
            'Price per Clip',
            'Marketing Cost',
            'Wire Inches',
            'Wire Cost',
            'Autoclipper Cost',
            'Processors',
            'Memory',
            'Trust',
            'Operations',
            'Creativity',
            'Riskiness',
            'Yomi',
            'Investment Engine Upgrade Cost',
            'Tournament Cost'
        ]
        self._avail_observations += [pname+" Activated" for pname in UP_PROJECT_IDS.keys()]
        self._avail_observations += [pname+" Available" for pname in UP_PROJECT_IDS.keys()]
        
        self._ac_avail_funcs = {
            'Do Nothing': lambda x: True,
            'Make Paperclip': lambda x: x['Wire Inches']>= 1,
            'Lower Price': lambda x: x['Price per Clip']>0.01,
            'Raise Price': lambda x: True,
            'Expand Marketing': lambda x: x['Available Funds']>=x['Marketing Cost'],
            'Buy Wire': lambda x: x['Available Funds']>=x['Wire Cost'],
            'Buy Autoclipper': lambda x: x['Available Funds']>=x['Autoclipper Cost'],
            'Add Processor': lambda x: x['Trust']>=x['Processors']+x['Memory'],
            'Add Memory': lambda x: x['Trust']>=x['Processors']+x['Memory'],
            'Set Investment Low': lambda x: x['Algorithmic Trading Activated'] and x['Riskiness'] != 7,
            'Set Investment Medium': lambda x: x['Algorithmic Trading Activated'] and x['Riskiness'] != 5,
            'Set Investment High': lambda x: x['Algorithmic Trading Activated'] and x['Riskiness'] != 1,
            'Set Investment Low': lambda x: x['Algorithmic Trading Activated'],
            'Set Investment Medium': lambda x: x['Algorithmic Trading Activated'],
            'Set Investment High': lambda x: x['Algorithmic Trading Activated'],
            'Withdraw': lambda x: x['Algorithmic Trading Activated'],
            'Deposit': lambda x: x['Algorithmic Trading Activated'],
            'Upgrade Investment Engine': lambda x: x['Algorithmic Trading Activated'] and x['Yomi']>=x['Investment Engine Upgrade Cost'],
            'Quantum Compute': lambda x: x['Quantum Computing Activated'] and x['Number of Photonic Chips'] != 0,
            'Buy MegaClipper': lambda x: x['Available Funds']>=x['MegaClipper Cost'] and x['MegaClippers Activated'],
            'Run New Tournament': lambda x: x['Strategic Modeling Activated'] and x['Operations']>=x['Tournament Cost']
        }
        
        # Web driver setup
        self._webdriver_name = webdriver_name
        self._webdriver_path = webdriver_path
        self._headless = headless
        self._setUpWebdriver()
        
        # Remaining setup
        self._url = url
        self._verbose = verbose
        self.reset()

    # "Public" functions
    def reset(self):
        try:
            self._driver.get(self._url)
        except:
            print("WARNING: Unable to fetch page on reset. Starting new driver.")
            self._setUpWebdriver()
            self.reset()
        self._active_projects = []
        self._acquired_buttons = {}
        self._updateGameState()
    
    def quit(self):
        self._driver.quit()
        self._driver.stop_client()
    
    def makeObservation(self, fields):
        self._updateGameState()
        observation = OrderedDict()
        visible_fields = []
        project_fields = []
        for field in fields:
            if field.endswith("Activated"):
                project_fields.append(field)
            else:
                visible_fields.append(field)
        for field in visible_fields:
            observation[field] = self._state.get(field)
        for field in project_fields:
            project_name = field[:-10]           
            if project_name in self._active_projects:
                observation[field] = 1
            else:
                observation[field] = 0
        return observation
    
    def takeAction(self, action_name):
        success = False
        # Buttons
        if action_name in self._all_buttons:
            try:
                success = self._clickButton(action_name)                
                # Register that project has been activated
                if success:
                    if action_name in self._project_buttons.keys():
                        project_name = action_name[9:]
                        self._active_projects.append(project_name)              
                    
            except http.client.RemoteDisconnected:
                print("ERROR: Disconnected while attempting action.")
                print("WARNING: Handling this by resetting. If this was in the middle of a sample path it's going to mess it up a lot.")
                self.reset()
                self.takeAction(action_name)
        elif action_name == "Do Nothing":
            pass      
        else:
            print("WARNING: Not sure what to do with action "+action_name+".")
            print("Doing nothing.")
        
        if self._verbose:
            if success:
                print("Took action "+action_name+"!")
            else:
                print("ERROR: Failed to take action "+action_name+"!")
        return success
    
    def save_screenshot(self, filename):
        self._driver.save_screenshot(filename)
    
    # "Private" functions
    def _setUpWebdriver(self):
        if self._webdriver_name == 'Chrome':            
            self._chrome_options = chrome.options.Options()
            if self._headless:
                self._chrome_options.add_argument("--headless")
            self._driver = webdriver.Chrome(chrome_options=self._chrome_options)
        elif self._webdriver_name == 'PhantomJS':
            if self._webdriver_path == None:
                try:
                    self._driver = webdriver.PhantomJS()
                except:
                    raise Exception("Failed to find phantomjs on system. Need to specify webdriver_path=/path/to/phantomjs.")
            else:
                self._driver = webdriver.PhantomJS(self._webdriver_path)
            if self._headless == False:
                print("WARNING: Specified not headless, but PhantomJS is headless only.")
        else:
            raise NotImplementedError("No implementation for webdriver {}.".format(self._webdriver_name))
       
    def _updateGameState(self):
        self._state = self._getGameStateFromPage()
    
    def _getGameStateFromPage(self):
        try:
            return UPGameState(self._driver)
        except http.client.RemoteDisconnected:
            print("ERROR: Disconnected while attempting to fetch game state.")
            print("WARNING: Handling this by resetting. If this was in the middle of a sample path it's going to mess it up a lot.")
            self.reset()
            return self._getGameStateFromPage()     
        
    def _findButton(self, name):
        if name in self._acquired_buttons.keys():
            button = self._acquired_buttons[name]
        else:
            try:
                button = self._driver.find_elements_by_id(self._all_buttons[name])[0]            
                self._acquired_buttons[name] = button
            except:
                if self._verbose:
                    print("WARNING: couldn't find button {}.".format(name))
                button = None
                clickable = False
        if button != None:
            try:
                clickable = not button.get_property('disabled')
            except selenium.common.exceptions.StaleElementReferenceException:
                button = None
                clickable = False
                if self._verbose:
                    print("WARNING: attempted to access missing/removed button {}.".format(name))
        return button, clickable
        
    def _clickButton(self, button_name):
        success = False
        actions = ActionChains(self._driver)
        button, clickable = self._findButton(button_name)
        if clickable:
            actions.click(button)
            actions.perform()
            success = True       
        return success
    
    def actionAvailable(self, ac_name, observation):
        # Action availability that can be resolved with other observables
        if ac_name in self._ac_avail_funcs.keys():
            return self._ac_avail_funcs[ac_name](observation)
        
        # Project activations
        elif ac_name in self._project_buttons.keys():
            project_name = ac_name[9:]
            if observation[project_name+' Available']:
                button, clickable = self._findButton(ac_name)
                return clickable
            else:
                return False
        
        # Other actions that are buttons
        elif ac_name in self._all_buttons.keys():
            button, clickable = self._findButton(ac_name)
            return clickable
        
        # Other actions 
        elif ac_name == "Do Nothing":
            return True
        else:
            raise Exception("Don't know how to determine availability of {}.".format(ac_name))
    
    def getAvailableActions(self, ac_names):
        observation = self.makeObservation(self._avail_observations)
        acs_avail = []
        for ac_name in ac_names:
            acs_avail.append(float(self.actionAvailable(ac_name, observation)))
        return acs_avail
