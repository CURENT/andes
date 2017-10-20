from dime import dime


localhost = 'tcp://127.0.0.1:5000'

dimec = dime.Dime('mod1', localhost)
dimec.start()

Event = dict()

Event['name'] = ['Line']
Event['id'] = [1]
Event['duration'] = [0.1]
Event['action'] = [1]
Event['time'] = [-1]


# Event['name'] = ['Line', 'Bus']
# Event['id'] = [1, 5]
# Event['duration'] = [0.1, 0.1]
# Event['action'] = [1, 0]
# Event['time'] = [2, 5]


if __name__ == '__main__':
    dimec.send_var('sim', 'Event', Event)
    dimec.exit()
