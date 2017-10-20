A bot that plays [Universal Paperclips](http://decisionproblem.com/paperclips/index2.html).

Intended for learning about reinforcement learning.

# Dependencies

## General
- Python 3

## Game Handler
- Python modules:
	- Selenium
	- lxml
- One of the following webdrivers:
	- Chromium
		- e.g. On Ubuntu: sudo apt install chromium-browser chromium-chromedriver
	- PhantomJS built with ghostdriver support
		- The binary distribution from http://phantomjs.org/ is best.

## Game Emulation
- Python modules:
	- py\_mini\_racer

## Learning
- Python modules:
	- Numpy
	- [OpenAI Gym](https://github.com/openai/gym)
	- [OpenAI Baselines](https://github.com/openai/baselines)

# Known issues

## Known issue 1
You get a message like:  
"""  
/yourpythonpath/lasagne/layers/pool.py", line 6, in <module>  
    from theano.tensor.signal import downsample  
ImportError: cannot import name 'downsample'  
"""  
See https://github.com/Theano/Theano/issues/4337. Fix is to change line 6 of pool.py to   
	"from theano.tensor.signal import pool"  
and change instance(s) of  
	"downsample.max_pool_2d"  
to  
	"pool.pool_2d".  
