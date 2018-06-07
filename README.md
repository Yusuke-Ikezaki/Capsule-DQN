# Capsule-DQN
DQN using Capsule Network.

## Installation
```
$ git clone https://github.com/Yusuke-Ikezaki/Capsule-DQN.git
```

## Set up
```
$ cd Capsule-DQN/
$ pip install -r requirements.txt
```

## How to use

### Simple ver
```
$ python main.py
```

### Custom ver
```
$ python main.py --PARAM1=VALUE1, --PARAM2=VALUE2, ...
```

Parameters are defined in config.py

#### Example
```
$ python main.py --episode=10000, --restore=False
```

## Option
If you want to check the learning status on your smartphone, please uncomment the line about [Hyperdash](https://hyperdash.io) in main.py  
