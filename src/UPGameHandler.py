from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

class UPGameState(object):
    _scalar_values = {
        'Paperclips': None,
        'Available Funds': None,
        'Unsold Inventory': None,
        'Price per Clip': None,
        'Public Demand': None,
        'Marketing Level': None,
        'Marketing Cost': None,
        'Manufacturing Clips per Second': None,
        'Wire Inches': None,
        'Wire Cost': None
    }
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
        'Wire Cost': ['id','wireCost']
    }
    
    def __init__(self, driver):        
        # Scalar values
        for field, finder in self._scalar_values_finders.items():
            element = driver.find_elements(by=finder[0], value=finder[1])[0]
            value = element.get_attribute('innerHTML')
            if value == None:
                self._scalar_values[field] = None
            else:
                try:
                    self._scalar_values[field] = float(value)
                except:
                    # Case where commas are used to separate thousands
                    self._scalar_values[field] = float(value.replace(",",""))
        
        # All kinds of values combined
        self._all_values = {}
        self._all_values.update(self._scalar_values)
        
    def get(self, field):
        return self._all_values[field]
    
    def __str__(self):
        return self._scalar_values.__str__()        

class UPGameHandler(object):
    def __init__(self, url, verbose=False):
        # Class constants
        self._all_buttons = {
            'Make Paperclip': 'btnMakePaperclip',
            'Lower Price': 'btnLowerPrice',
            'Raise Price': 'btnRaisePrice',
            'Expand Marketing': 'btnExpandMarketing',
            'Buy Wire': 'btnBuyWire',
            'Buy Autoclipper': 'btnMakeClipper'
        }
        self._all_actions = list(self._all_buttons.keys())
        
        # Setup
        self._driver = webdriver.Chrome()
        self._url = url
        self._verbose = verbose
        self._driver.get(url)
        self._updateGameState()

    # "Public" functions
    def reset(self):
        self._driver.get(self._url)
        self._updateGameState()
    
    def makeObservation(self, fields):
        self._updateGameState()
        observation = {}
        for field in fields:
            observation[field] = self._state.get(field)
        return observation
    
    def takeAction(self, action_name):
        success = False
        if action_name in self._all_buttons:
            success = self._clickButton(action_name)
        else:
            print("Not sure what to do with action "+action_name+".")
        
        if self._verbose:
            if success:
                print("Took action "+action_name+"!")
            else:
                print("ERROR: Failed to take action "+action_name+"!")
        return success
    
    # "Private" functions    
    def _updateGameState(self):
        self._state = self._getGameStateFromPage()
    
    def _getGameStateFromPage(self):
        return UPGameState(self._driver)
        
    def _findButton(self, action_name):
        button = self._driver.find_elements_by_id(self._all_buttons[action_name])[0]
        clickable = not button.get_property('disabled')
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
    
    def _getAvailableActions(self):
        available = []
        for button_name in self._all_buttons:
            button, clickable = self._findButton(button_name)
            if clickable:
                available.append(button_name)            
        return available
