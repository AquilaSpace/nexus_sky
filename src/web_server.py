# api-level accessor to the graph db
import sys, os
import logging

from flask import Flask, jsonify, json, abort
from functools import wraps
from src.app import app

class WebServer:
    def __init__(self):
        self.app = app

    def start(self):
        with app.app_context():
            self.app.run(port=6000)

if __name__ == '__main__': 
    WebServer().start()