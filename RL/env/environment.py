__version__ = '1.0'
__authors__ = 'Sihem Ouahouah & Miloud Bagaa'
__author_emails__ = 'sihem.ouahouah@aalto.fi & miloud.bagaa@aalto.fi'

class Environment(object):

    def __init__(self, areaSideSize = ENV_SIZE, observableAccessPoints = OBSERVABLE_ACCESS_POINTS, observableEvents = OBSERVABLE_EVENTS):
        self.render = False
        self.reset()


    def set_render(self):
        self.render = True

    def unset_render(self):
        self.render = False

    def reset(self):
        self.steps = 0
        self.observation_space = self.generate_state()
        self.nbCollision = 0

        return self.observation_space


    def step(self, action_agent):
        state = 0
        reward = 0
        done = False
        info = []

        return state, reward, done, info

    def generate_state(self):
        state = []
        return state