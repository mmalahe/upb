from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import selenium.common.exceptions
import selenium.webdriver.chrome as chrome
import lxml.html
import http
from collections import OrderedDict
import os

LOCAL_GAME_URL_TRAIN = "file://"+os.path.join(os.path.dirname(os.path.abspath(__file__)),"index2_train.html")
LOCAL_GAME_URL_STANDARD = "file://"+os.path.join(os.path.dirname(os.path.abspath(__file__)),"index2.html") 

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
        'Operations': ['id','operations']
    }
    
    def __init__(self, driver):        
        # The soup
        html_source = driver.find_element_by_xpath("//*").get_attribute("outerHTML")        
        html_parser = lxml.html.document_fromstring(html_source)
        
        # Scalar values
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
        
        # All kinds of values combined
        self._all_values = {}
        self._all_values.update(self._scalar_values)
        
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
            'Add Memory': 'btnAddMemory'
        }
        self._all_actions = list(self._all_buttons.keys())
        
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
        self._acquired_buttons = {}
        self._updateGameState()
    
    def quit(self):
        self._driver.quit()
        self._driver.stop_client()
    
    def makeObservation(self, fields):
        self._updateGameState()
        observation = OrderedDict()
        for field in fields:
            observation[field] = self._state.get(field)
        return observation
    
    def takeAction(self, action_name):
        success = False
        if action_name in self._all_buttons:
            try:
                success = self._clickButton(action_name)
            except http.client.RemoteDisconnected:
                print("ERROR: Disconnected while attempting action.")
                print("WARNING: Handling this by resetting. If this was in the middle of a sample path it's going to mess it up a lot.")
                self.reset()
                self.takeAction(action_name)
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
            button = self._driver.find_elements_by_id(self._all_buttons[name])[0]
            self._acquired_buttons[name] = button
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
