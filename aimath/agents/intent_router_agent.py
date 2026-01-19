from agno.agent import Agent
from agno.models.groq import Groq
from aimath.config.settings import Settings

class IntentRouterAgent:
    """
    Router Agent.
    Responsibilities:
    1. Classify the Math Domain (Algebra, Calculus, Geometry, Probability, Linear Algebra, etc.)
    2. Determine Solvability (Simple Arithmetic vs Complex Proof vs Word Problem)
    """

    def __init__(self):
        self.agent = Agent(
            model=Groq(id=Settings.LLM_MODEL),
            description="You are a routing system that classifies math problems.",
            instructions=[
                "Analyze the verified problem text.",
                "Output ONLY a single string token indicating the primary category.",
                "Categories: ALGEBRA, CALCULUS, GEOMETRY, PROBABILITY, LINEAR_ALGEBRA, STATISTICS, ARITHMETIC, OTHER.",
                "Do NOT output sentences."
            ]
        )

    def route(self, problem_text: str) -> str:
        response = self.agent.run(f"Classify this problem: {problem_text}")
        return response.content.strip().upper()
