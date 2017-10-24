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

## Plotting
- Python modules:
	- matplotlib

## Documentation
- Python modules:
	- Sphinx

# Installation
~~~~
git clone https://github.com/mmalahe/upb.git
cd upb
pip install -e .
~~~~
