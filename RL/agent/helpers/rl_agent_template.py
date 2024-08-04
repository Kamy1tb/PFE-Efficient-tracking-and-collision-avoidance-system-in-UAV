from abc import ABC


class RLAgent(ABC):

    def configure(self, params=None):
        pass

    def store_transition(self, state=None, action=None, next_state=None, reward=None, done=None):
        pass

    def step_learn(self, updates=1):
        pass

    def episode_learn(self):
        pass

    def save_model(self, path=""):
        pass

    def load_model(self, path=""):
        pass

    def save_output(self, path=""):
        pass

    def update_learn_params(self):
        pass

    def choose_action(self, state):
        pass