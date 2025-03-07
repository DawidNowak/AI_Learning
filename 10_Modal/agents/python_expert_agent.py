import modal
from agents.agent import Agent


class PythonExpertAgent(Agent):
    """
    An Agent that runs LLama3.1 that's running remotely on Modal
    """

    name = "Python Expert Agent"
    color = Agent.RED

    def __init__(self):
        """
        Set up Agent by creating an instance of the modal class
        """
        self.log(f"{self.name} is initializing - connecting to modal")
        Expert = modal.Cls.from_name("python-expert", "Expert")
        self.expert = Expert()
        self.log(f"{self.name} is ready")
        
    def explain(self, description: str) -> str:
        """
        Make a remote call to explain this code
        """
        self.log(f"{self.name} is calling remote model")
        result = self.expert.explain.remote(description)
        self.log(f"{self.name} completed - explanation: ${result}")
        return result
