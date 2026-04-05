#!/bin/bash

0 9 * * 1 python ~/toollib/toollib/datasource_monitor/lark/main.py >> ~/datasource_monitor.log 2>&1