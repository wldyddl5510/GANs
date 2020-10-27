from utils.loader_lists import AGENTS
import agents

"""
    class to load agents
"""

class AgentLoader:


    def __init__(self, args, module, env, logger = None, log_dir = None):
        if args.agent not in [*AGENTS]:
            raise NotImplementedError

        self.logger = logger

        # load agent
        AgentClass = AGENTS[args.agent]
        self.agent = AgentClass(args, module, env, logger, log_dir)

        if logger is not None:
            self.logger.info("Sucessfully loaded agent: " + args.agent)
