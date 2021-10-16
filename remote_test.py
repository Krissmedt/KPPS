# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:24:20 2020

@author: KristofferSmedt
"""

import time
import sys

run_minutes = 10


print(time.strftime("%H:%M:%S"))
for minute in range(0,run_minutes-1):
    time.sleep(60)
    print(time.strftime("%H:%M:%S"))
