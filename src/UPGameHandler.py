from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


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
        'Make Paperclip': 'btnMakePaperclip',
        'Lower Price': 'btnLowerPrice',
        'Raise Price': 'btnRaisePrice',
        'Expand Marketing': 'btnExpandMarketing',
        'Buy Wire': 'btnBuyWire'
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
        button, clickable = self._findButton(action_name)
        if clickable:
            actions.click(button)
            actions.perform()
            success = True       
        if not success:
            print "ERROR: Failed to take action "+action_name+"!"
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

gh = UPGameHandler("http://www.decisionproblem.com/paperclips/index2.html")
print gh.getState()
gh.takeAction('Make Paperclip')
print gh.getState()
