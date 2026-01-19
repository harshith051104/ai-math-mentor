from typing import Dict, Any, List
from agno.agent import Agent
from agno.models.groq import Groq
from aimath.config.settings import Settings
import json
import re

class ExplainerAgent:
    """
    Enhanced JEE-focused Explainer Agent.
    Provides comprehensive, pedagogical explanations.
    """
    
    def __init__(self):
        self.agent = Agent(
            model=Groq(id=Settings.LLM_MODEL),
            description="You are an expert JEE Mathematics Tutor.",
            instructions=[
                "ROLE: You are a compassionate, expert JEE Mathematics tutor.",
                "",
                "TASK: Provide a complete, pedagogical explanation of the solution.",
                "",
                "OUTPUT STRUCTURE (MANDATORY):",
                "You must output a structured explanation with these sections:",
                "",
                "1. CONCEPT & FORMULA",
                "   - State the topic/chapter (e.g., 'Quadratic Equations', 'Trigonometry')",
                "   - List 2-3 key formulas or concepts used",
                "   - Keep it concise (max 3 lines)",
                "",
                "2. SOLUTION STRATEGY",
                "   - Briefly state the approach (e.g., 'Factorization', 'Substitution')",
                "   - Explain WHY this method was chosen",
                "   - 1-2 sentences only",
                "",
                "3. KEY INSIGHT",
                "   - What's the 'aha!' moment in this problem?",
                "   - What pattern recognition or trick made it easy?",
                "   - Example: 'The key was recognizing 123456 = 64 × 1929'",
                "",
                "4. LEARNING POINTS",
                "   - 2-3 bullet points of what student should remember",
                "   - Focus on transferable skills, not just this problem",
                "",
                "5. COMMON MISTAKES (if applicable)",
                "   - What errors do students typically make here?",
                "   - 1-2 specific examples",
                "",
                "TONE:",
                "- Warm, encouraging, patient",
                "- Use 'we' and 'let's' (inclusive)",
                "- Explain like you're sitting next to the student",
                "- Avoid condescension or over-simplification",
                "",
                "LENGTH:",
                "- Total: 150-250 words",
                "- Each section: 2-4 sentences max",
                "",
                "LATEX:",
                "- Use LaTeX for all math: $x^2$, $\\sqrt{2}$, $\\frac{a}{b}$",
                "- Use \\cdot for multiplication, not *",
                "",
                "EXAMPLES:",
                "Good: 'The key insight was factoring out the perfect square $8^2$ from $123456$.'",
                "Bad: 'We simplified the expression under the square root.'",
                "",
                "OUTPUT FORMAT:",
                "Return as JSON with keys:",
                "{ ",
                "  'concept': 'Topic name',",
                "  'strategy': 'Brief strategy',",
                "  'key_insight': 'The aha moment',",
                "  'learning_points': ['Point 1', 'Point 2', ...],",
                "  'common_mistakes': ['Mistake 1', 'Mistake 2'],",
                "  'difficulty': 'Easy|Medium|Hard',",
                "  'jee_relevance': 'JEE Main|JEE Advanced|Both'",
                "}"
            ],
            markdown=False
        )
    
    def explain(
        self, 
        problem_text: str, 
        solution_steps: List[str], 
        final_answer: str,
        category: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive JEE-focused explanation.
        """
        prompt = f"""
Problem: {problem_text}
Category: {category}
Steps: {solution_steps}
Final Answer: {final_answer}

Generate a structured JEE-style explanation following the output format.
"""
        
        response = self.agent.run(prompt)
        
        # Parse JSON response
        try:
            content = response.content
            # Extract JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                explanation_data = json.loads(match.group(0))
            else:
                # Fallback
                explanation_data = {
                    "concept": category,
                    "strategy": "Direct computation",
                    "key_insight": content[:200],
                    "learning_points": ["Review the solution carefully"],
                    "common_mistakes": [],
                    "difficulty": "Medium",
                    "jee_relevance": "JEE Main"
                }
            
            return explanation_data
            
        except Exception as e:
            # Graceful fallback
            return {
                "concept": category,
                "strategy": "Standard approach",
                "key_insight": response.content[:300],
                "learning_points": ["Focus on the methodology"],
                "common_mistakes": [],
                "difficulty": "Medium",
                "jee_relevance": "JEE Main",
                "error": str(e)
            }
