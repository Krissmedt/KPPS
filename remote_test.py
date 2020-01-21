# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:24:20 2020

@author: KristofferSmedt
"""

import time
import sys

run_minutes = 2



for minute in range(0,run_minutes):
    print(time.strftime("%H:%M:%S"))
    time.sleep(60)