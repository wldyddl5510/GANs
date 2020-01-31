from settings import AGENTS
import pdb

"""
    class to load agents
"""

class AgentLoader:


    def __init__(self, args, module, env, logger = None):
        if args.agent not in [*AGENTS]:
            raise NotImplementedError

        # load agent
        AgentClass = AGENTS[args.agent]
        self.agent = AgentClass(args, module, env, logger)

