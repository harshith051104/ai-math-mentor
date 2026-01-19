
from typing import Dict, Any
import json
import re
from agno.agent import Agent
from agno.models.groq import Groq
from aimath.config.settings import Settings

class GuardrailAgent:
    """
    Mathematical Safety Guardrail.
    Inspects the proposed solution plan for unwanted algebraic risks (degree explosion, illegal division).
    """
    
    def __init__(self):
        self.agent = Agent(
            model=Groq(id=Settings.LLM_MODEL),
            description="You are a Mathematical Safety Guardrail.",
            instructions=[
                "ROLE: You are a Mathematical Safety Guardrail.",
                "YOU DO NOT SOLVE THE PROBLEM.",
                "YOU DO NOT SUGGEST NEW METHODS.",
                "YOU DO NOT EXPLAIN.",
                "",
                "TASK:",
                "Inspect the proposed solution plan and decide whether it is",
                "SAFE to execute under JEE-level mathematical discipline.",
                "",
                "INPUT:",
                "- problem_text",
                "- solution_plan",
                "",
                "CHECK FOR VIOLATIONS USING THE RULES BELOW.",
                "",
                "🚨 Guardrail Rules (CRITICAL)",
                "🔴 Rule 1: Illegal Division / Cancellation",
                "FAIL if the plan:",
                "- Divides by expressions containing variables",
                "- Cancels expressions like (1+y), (x−y)",
                "- Does so without explicitly stating domain restrictions",
                "",
                "🔴 Rule 2: Composite Variable Solving",
                "FAIL if the plan:",
                "- Solves for xy, x/y, x−y, etc.",
                "- Uses them as primary substitution targets",
                "",
                "🔴 Rule 3: Polynomial Degree Explosion",
                "FAIL if:",
                "- Original problem is degree ≤ 2",
                "- Plan introduces degree ≥ 3 algebra (especially quartics)",
                "",
                "🔴 Rule 4: Rational Substitution in Polynomial Systems",
                "FAIL if:",
                "- Problem is polynomial",
                "- Plan introduces rational expressions unnecessarily",
                "",
                "🔴 Rule 5: Planner Doing Algebra",
                "FAIL if the plan itself contains:",
                "- explicit equations",
                "- algebraic simplifications",
                "- expansions",
                "- Planner must describe actions, not math.",
                "",
                "🟡 Rule 6: Missed JEE Heuristics (Soft Fail)",
                "FLAG (but not hard fail) if:",
                "- Integer testing is ignored when constants are small",
                "- Symmetry is obvious but unused",
                "This results in: recommended_action = 'SWITCH_STRATEGY'",
                "",
                "🔴 Rule 7: Arithmetic Delegation (CRITICAL)",
                "FAIL if:",
                "- Problem type is ARITHMETIC",
                "- Plan does NOT explicitly state 'Use Calculator Tool'",
                "- Plan contains words: 'approximate', 'estimate', 'roughly', 'about'",
                "- Plan suggests manual calculation for numbers > 100",
                "",
                "PASS ONLY if:",
                "- Plan says: 'Delegate to Calculator' or 'Use SymPy Tool'",
                "",
                "🔴 Rule 8: No Mental Arithmetic for Large Numbers",
                "FAIL if:",
                "- Problem involves numbers with 4+ digits",
                "- Plan describes step-by-step arithmetic (e.g., 'carry the 1')",
                "- Expected output: recommended_action = 'USE_TOOL'",
                "",
                "OUTPUT:",
                "Return STRICT JSON ONLY.",
                "NO markdown. NO extra text.",
                "",
                "SCHEMA:",
                "{",
                "  'is_safe': boolean,",
                "  'violation_type': 'string | null',",
                "  'reason': 'string | null',",
                "  'recommended_action': 'EXECUTE | SWITCH_STRATEGY | TRIGGER_HITL'",
                "}"
            ],
            markdown=False
        )

    def check(self, problem_text: str, plan: str) -> Dict[str, Any]:
        """
        Run the guardrail check on a plan.
        """
        prompt = f"Problem: {problem_text}\n\nProposed Plan:\n{plan}"
        response = self.agent.run(prompt)
        
        try:
            content_str = response.content
            # Robust JSON extraction
            match = re.search(r"```json\s*(.*?)```", content_str, re.DOTALL)
            if match:
                content_str = match.group(1).strip()
            else:
                start = content_str.find('{')
                end = content_str.rfind('}') + 1
                if start != -1 and end != 0:
                    content_str = content_str[start:end]
            
            return json.loads(content_str)
        except Exception as e:
            # If guardrail breaks, fail safe -> TRIGGER HITL
            return {
                "is_safe": False,
                "violation_type": "Guardrail_Error",
                "reason": f"Guardrail failed to parse confirmation: {str(e)}",
                "recommended_action": "TRIGGER_HITL"
            }
