from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

class FixedDict:
    def __init__(self, dictionary):
        self._dictionary = dictionary
    def __setitem__(self, key, item):
            if key not in self._dictionary:
                raise KeyError("The key {} is not defined.".format(key))
            self._dictionary[key] = item
    def __getitem__(self, key):
        return self._dictionary[key]

class UPGameState:
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
        for field, finder in self._scalar_values_finders.iteritems():
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
                        
    def __str__(self):
        return self._scalar_values.__str__()        

class UPGameHandler:
    _all_buttons = {
        'Make Paperclip': 'btnMakePaperclip'
    }
    _driver = None
    _state = None
    def __init__(self, url):
        self._driver = webdriver.Chrome()
        self._driver.get(url)
        self._updateGameState()

    # "Public" functions
    def getState(self):
        self._updateGameState()
        return self._state
        
    def takeAction(self, action_name):
        success = False
        actions = ActionChains(self._driver)
        button = self._findButton(action_name)
        actions.click(button)
        try:
            actions.perform()
            success = True
        except:
            print "ERROR: Failed to take action "+action_name+"!"
            success = False
        return success
    
    # "Private" functions
    def _updateGameState(self):
        self._state = self._getGameStateFromPage()
    
    def _getGameStateFromPage(self):
        return UPGameState(self._driver)
        
    def _findButton(self, action_name):
        return self._driver.find_elements_by_id(self._all_buttons[action_name])[0]

gh = UPGameHandler("http://www.decisionproblem.com/paperclips/index2.html")
print gh.getState()
gh.takeAction('Make Paperclip')
print gh.getState()
