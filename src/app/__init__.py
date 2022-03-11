# -*- coding: utf-8 -*-

from flask import Flask
from src.app.routes.satellite import satellite

app = Flask(__name__)

app.register_blueprint(satellite)