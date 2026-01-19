
import json
import re
from typing import Dict, Any
from agno.agent import Agent
from agno.models.groq import Groq
from aimath.config.settings import Settings

class VerifierAgent:
    def __init__(self):
        self.agent = Agent(
            model=Groq(id=Settings.LLM_MODEL),
            description="You are a Board Examiner.",
            instructions=[
                "ROLE: You are a deterministic JSON validator.",
                "You do NOT explain.",
                "You do NOT think out loud.",
                "",
                "CRITICAL VERIFICATION RULES:",
                "1. If solution used SymPy Calculator (tool_used='calculator'):",
                "   → Automatically mark as CORRECT unless obvious error",
                "   → SymPy is deterministic and trustworthy",
                "",
                "2. Verify numerical consistency:",
                "   - If symbolic form exists, check if decimal matches",
                "   - Allow small floating-point errors (< 0.001%)",
                "   - DO NOT flag 'clean' decimals as suspicious",
                "",
                "3. Common false positives to IGNORE:",
                "   - Repeating patterns like 11111.111 (can be valid!)",
                "   - 'Clean' looking decimals (e.g., 351.36...)",
                "   - Perfect divisions",
                "",
                "4. ONLY mark as incorrect if:",
                "   - Algebraic manipulation is clearly wrong",
                "   - Domain violation (e.g., √-1 without complex)",
                "   - Dimensional mismatch",
                "   - Division by zero",
                "",
                "Rules:",
                "1. If ANY step is invalid → is_correct = false.",
                "2. If division/cancellation occurs without domain check → false.",
                "3. If tool_used='calculator' and no obvious errors → true.",
                "4. If unsure → false.",
                "",
                "Output RAW JSON ONLY.",
                "Schema:",
                "{",
                "  'is_correct': boolean,",
                "  'critique': 'Specific concise feedback or Correct',",
                "  'adjusted_confidence': float,",
                "  'verification_method': 'symbolic|numerical|both'",
                "}"
            ],
            markdown=False
        )

    def verify(self, problem_text: str, solution_data: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        # SPECIAL CASE: Calculator solutions are auto-verified
        if solution_data.get("tool_used") == "calculator":
            # Perform basic sanity check only
            if self._sanity_check_calculator(solution_data):
                return {
                    "is_correct": True,
                    "critique": "Correct (verified by SymPy calculator)",
                    "adjusted_confidence": 1.0,
                    "verification_method": "symbolic"
                }
        
        # For LLM-based solutions, use the agent
        solution_str = json.dumps(solution_data)
        prompt = f"Problem: {problem_text}\nSolution: {solution_str}\nVerify this."
        
        if strict:
            prompt = "STRICT AUDIT MODE ENABLED.\n" + \
                     "1. CHECK EVERY SINGLE ALGEBRAIC STEP.\n" + \
                     "2. FAIL IF ANY DOMAIN RESTRICTION IS MISSED.\n" + \
                     "3. FAIL IF PLAN DOES NOT MATCH EXECUTION.\n" + \
                     prompt
                     
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
            return {
                "is_correct": False,
                "critique": f"JSON parsing failed. Error: {str(e)}. Raw response: {response.content}",
                "adjusted_confidence": 0.0
            }
    
    def _sanity_check_calculator(self, solution_data: Dict[str, Any]) -> bool:
        """
        Basic sanity check for calculator results.
        Only flag obvious errors like NaN, Infinity, or missing answer.
        """
        final_answer = solution_data.get("final_answer")
        
        # Check 1: Answer exists
        if final_answer is None or final_answer == "":
            return False
        
        # Check 2: Not NaN or Infinity
        try:
            num_val = float(final_answer)
            if not (-1e100 < num_val < 1e100):  # Reasonable range
                return False
        except:
            # Symbolic answer (like "3*sqrt(5)") is OK
            pass
        
        # Check 3: Steps exist
        if not solution_data.get("steps"):
            return False
        
        # Passed all checks
        return True
