"""
configuration file
"""

import logging
import datetime

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
now = datetime.datetime.now()
today = '%s-%s-%s'%(now.year, now.month, now.day)
current_time = '%s-%s-%s'%(now.hour, now.minute, now.second)
folder_path = 'log/' + today
