[![Build Status](https://travis-ci.com/m3ttiw/orange_cb_recsys.svg?branch=master)](https://travis-ci.com/m3ttiw/orange_cb_recsys)&nbsp;&nbsp;[![Coverage Status](https://coveralls.io/repos/github/m3ttiw/orange_cb_recsys/badge.png?branch=master)](https://coveralls.io/github/m3ttiw/orange_cb_recsys?branch=master)&nbsp;&nbsp;![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/rbarile17/framework_dependencies)&nbsp;&nbsp;[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/)

# Orange_cb_recsys

Framework for content-based recommender system

Installation
=============
``
pip install orange-cb-recsys
``

[PyLucene](https://lucene.apache.org/pylucene/) is required and will not be installed like other dependencies, you will need to install it personally.

Usage
=====
There are two types of use for this framework
It can be used through API or through the use of a config file

API Usage
---------
The use through API is the classic use of a library, classes and methods are used by invoking them.

Example: 

![Example](img/run.PNG)


Config Usage
------------
The use through the config file is an automated use.

Just indicate which algorithms you want to use and change variables where necessary without having to call classes or methods
This use is intended for users who want to use many framework features.

Examples:

![Example](img/item.PNG)
![Example](img/rating.PNG)

there are more example in the directory: 

``
orange_cb_recsys/content_analyzer/ratings_manager/
``