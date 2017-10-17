from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import selenium.common.exceptions
import selenium.webdriver.chrome as chrome
import lxml.html

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
        'Wire Cost': None,
        'Autoclipper Cost': None,
        'Number of Autoclippers': None
        
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
        'Wire Cost': ['id','wireCost'],
        'Autoclipper Cost': ['id','clipperCost'],
        'Number of Autoclippers': ['id','clipmakerLevel2']
    }
    _derived_scalar_values = {
        'Autoclipper Purchasable': None
    }
    
    def __init__(self, driver):        
        # The soup
        html_source = driver.find_element_by_xpath("//*").get_attribute("outerHTML")        
        html_parser = lxml.html.document_fromstring(html_source)
        
        # Scalar values
        for field, finder in self._scalar_values_finders.items():
            # Find value         
            if finder[0] == 'id':
                #~ value = soup.find(id=finder[1]).contents[0]
                value = html_parser.get_element_by_id(finder[1]).text_content()
            else:
                raise NotImplementedError("Don't know how to find things by "+finder[0])
            
            # Convert value
            if value == None:
                self._scalar_values[field] = None
            else:
                try:
                    self._scalar_values[field] = float(value)
                except:
                    # Cases where html spaces have been used
                    value = value.replace("&nbsp;","")
                    
                    # Case where commas are used to separate thousands.
                    # Sometimes they're used for decimals, but that scaling
                    # should be captured by any learning system regardless
                    value = value.replace(",","")
                    value = value.replace(u'\xa0',"")
                    self._scalar_values[field] = float(value)
                    
                    # Special cases where the exact value is actually important
                    if field == 'Autoclipper Cost':
                        self._scalar_values[field] /= 100.0
        
        # Derived values
        self._derived_scalar_values['Autoclipper Purchasable'] = 1.0*(self._scalar_values['Available Funds'] >= self._scalar_values['Autoclipper Cost'])
        
        # All kinds of values combined
        self._all_values = {}
        self._all_values.update(self._scalar_values)
        self._all_values.update(self._derived_scalar_values)
        
    def get(self, field):
        return self._all_values[field]
    
    def __str__(self):
        return self._scalar_values.__str__()        

class UPGameHandler(object):
    def __init__(self, url, selenium_executor=None, verbose=False, headless=True):
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
        
        # Web driver setup
        chrome_options = chrome.options.Options()
        if headless:
            chrome_options.add_argument("--headless")
        self._selenium_executor = selenium_executor
        if self._selenium_executor == None:
            self._chrome_options = chrome_options
            self._driver = webdriver.PhantomJS("/home/mikl/sfw/phantomjs-2.1.1-linux-x86_64/bin/phantomjs")
            #~ self._driver = webdriver.Chrome(chrome_options=self._chrome_options)
        else:
            self._driver_capabilities = chrome_options.to_capabilities()
            self._driver = webdriver.Remote(command_executor=self._selenium_executor, desired_capabilities=self._driver_capabilities)
        
        # Remaining setup
        self._url = url
        self._verbose = verbose
        self._driver.get(self._url)
        self._updateGameState()

    # "Public" functions
    def reset(self):
        if self._selenium_executor == None:
            #~ self._driver = webdriver.PhantomJS()
            self._driver = webdriver.PhantomJS("/home/mikl/sfw/phantomjs-2.1.1-linux-x86_64/bin/phantomjs")
        else:
            self._driver = webdriver.Remote(command_executor=self._selenium_executor, desired_capabilities=self._driver_capabilities)
        self._acquired_buttons = {}
        self._driver.get(self._url)
        self._updateGameState()
    
    def quit(self):
        try:
            self._driver.quit()
            self._driver.stop_client()
        except selenium.common.exceptions.WebDriverException:
            print("Session already closed.")
    
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
        
    def _findButton(self, name):
        if name in self._acquired_buttons.keys():
            button = self._acquired_buttons[name]
        else:
            button = self._driver.find_elements_by_id(self._all_buttons[name])[0]
            self._acquired_buttons[name] = button
        clickable = not button.get_property('disabled')
        return button
        
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
