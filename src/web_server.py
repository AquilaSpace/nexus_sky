# api-level accessor to the graph db
import sys, os
import logging

from flask import Flask, jsonify, json, abort
from functools import wraps
from src.app import app
from config import config
from apscheduler.schedulers.background import BackgroundScheduler
from src.db import init_db_connection

class WebServer:
    def __init__(self):
        self.app = app

    def start(self):
        self.app.run(debug=config['debug'], port=6000)

if __name__ == '__main__': 
    WebServer().start()