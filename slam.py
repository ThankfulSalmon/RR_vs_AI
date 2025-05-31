from transitions import Machine

class SLAMSystem:
    states = ['initializing', 'mapping', 'tracking', 'lost']

    def __init__(self):
        self.machine = Machine(model=self, states=SLAMSystem.states, initial='initializing')
        self.machine.add_transition('start_mapping', 'initializing', 'mapping')
        self.machine.add_transition('start_tracking', 'mapping', 'tracking')
        self.machine.add_transition('lose_tracking', 'tracking', 'lost')
        self.machine.add_transition('recover_tracking', 'lost', 'tracking')
        self.machine.add_transition('reset', '*', 'initializing')

