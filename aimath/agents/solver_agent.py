from typing import Dict, Any, List
import re
import json
from agno.agent import Agent
from agno.models.groq import Groq
from aimath.config.settings import Settings
from aimath.database.vector_store import VectorStore
from aimath.agents.guardrail_agent import GuardrailAgent

from aimath.tools.calculator import Calculator

class SolverAgent:
    """
    Solver Agent.
    Responsibilities:
    1. Retrieve similar problems/formulas from Knowledge Base (RAG) [Bounded].
    2. Formulate a Step-by-Step Plan (Symbolic Strategy).
    3. VALIDATE Plan with Guardrail Agent.
    4. Execute the plan to get a final answer (Delimited Text Output).
    5. Return confidence score.
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.guardrail = GuardrailAgent()
        self.calculator = Calculator()
        
        self.planner = Agent(
            model=Groq(id=Settings.LLM_MODEL),
            description="You are a math planning expert.",
            instructions=[
                "You are a JEE Mathematics Strategy Expert.",
                "Objective: Create a surgical, mechanical execution plan.",
                "Rules:",
                "1. **Conciseness**: The plan must be short and actionable. Do NOT explain concepts.",
                "2. **Sign Check**: Be extremely careful with signs (e.g., Vieta's sum is $-b/a$).",
                r"3. **Variables**: List ALL variables with domains (e.g., $x \in \mathbb{R}^+$).",
                r"4. **Recurrence**: If sequence, explicitly state $a_{n} = f(a_{n-1})$.",
                "5. **No Calculations**: Describe ONLY logical actions. Do NOT include equations or algebraic expressions.",
                "6. **ABSOLUTE PROHIBITION**: Do NOT do algebra in the plan.",
                "   - Do NOT solve for composite terms like $xy$.",
                "   - Do NOT divide by variable expressions in the plan.",
                "   - Do NOT introduce rational expressions.",
                "   - The plan must preserve polynomial structure.",
                "   - Example: 'Rearrange first equation' is GOOD. 'x = 7+y' is BAD.",
                "Output Format:",
                "**Concept:** <Brief concept name>",
                "**Strategy:**",
                "1. <Step 1>",
                "2. <Step 2>"
            ],
            markdown=False
        )
        
        self.executor = Agent(
            model=Groq(id=Settings.LLM_MODEL),
            description="You are a precise math engine.",
            instructions=[
                "ROLE: You are a JEE Mathematics Solver.",
                "PRIMARY GOAL: Solve the given problem correctly, rigorously, and efficiently, exactly as expected in a JEE examination.",
                "",
                "GENERAL RULES (NON-NEGOTIABLE):",
                "1. You must NEVER guess.",
                "2. You must NEVER handwave or say 'use a calculator' and still give an answer.",
                "3. If you are unsure at any point, STOP and report uncertainty.",
                "",
                "STRATEGY RULES:",
                "4. Preserve the mathematical structure of the problem.",
                "   - If the problem is polynomial, keep it polynomial.",
                "   - Do NOT introduce unnecessary fractions or rational expressions.",
                "5. Do NOT solve for composite expressions like xy, x-y, x/y unless explicitly required.",
                "6. Do NOT divide or cancel expressions involving variables unless the cancellation is fully justified and domain restrictions are stated.",
                "",
                "ARITHMETIC RULE (CRITICAL):",
                "7. If the problem is purely arithmetic (e.g., large multiplications), request or defer to a tool. Do NOT compute numerically if unsafe.",
                "",
                "DEGREE & COMPLEXITY CONTROL:",
                "8. If algebraic manipulation increases the degree beyond what is expected (e.g., quadratic -> quartic), STOP and switch strategy.",
                "   Prefer: integer testing, factor inspection, symmetry, substitution of small values.",
                "",
                "EXECUTION DISCIPLINE:",
                "9. Show only mathematically meaningful steps.",
                "10. Avoid narration such as 'further simplified' or 'expanded'.",
                r"11. Every division or cancellation must include its validity condition (e.g., $y \neq -1$).",
                r"12. LaTeX Formatting: Use LaTeX for ALL math expressions (e.g., $x^2 + y^2 = r^2$).",
                "",
                "ABSOLUTE PROHIBITIONS:",
                "- No guessing",
                "- No illegal cancellation",
                "- No arithmetic estimation by intuition",
                "- No unjustified shortcuts",
                "",
                "OUTPUT REQUIREMENTS:",
                "1. Produce a clean, exam-ready solution.",
                "2. End with a clear FINAL ANSWER.",
                "3. Provide a confidence score that reflects correctness honestly.",
                "",
                "Output Format:",
                "Do NOT output JSON.",
                "Use this exact text format:",
                "---STEPS---",
                "1. First step here",
                "2. Second step here",
                "---FINAL_ANSWER---",
                "The final result",
                "---CONFIDENCE---",
                "1.0"
            ],
            markdown=False
        )

    def solve(self, problem_text: str, problem_type: str = "GENERAL", context_str: str = "") -> Dict[str, Any]:
        """
        Solve the problem.
        If problem_type is ARITHMETIC, routes to deterministic tool.
        """
        
        # 🔥 CRITICAL: Route arithmetic BEFORE LLM planning
        if problem_type == "ARITHMETIC":
            return self._solve_with_tool(problem_text)

        # 1. RAG Retrieval (Bounded)
        if not context_str:
            results = self.vector_store.query(problem_text, n_results=1) # Limit to Top-1
            # Flatten results (Chroma returns list of lists)
            docs = results.get('documents', [[]])[0]
            if docs:
                 # Truncate to prevent token overflow (approx 1500 chars)
                context_str = docs[0][:1500]
            else:
                context_str = ""

        # 2. Planning (No Context to avoid hallucination form mismatched examples)
        plan_response = self.planner.run(f"Problem: {problem_text}\nCreate a strategy plan.")
        plan = plan_response.content
        
        # Guard: LLM-based Safety Check
        guard_result = self.guardrail.check(problem_text, plan)
        
        if not guard_result.get("is_safe", False):
            # Guardrail intercepted!
             return {
                "plan": plan,
                "steps": [],
                "final_answer": "Guardrail Intercepted",
                "confidence": 0.0,
                "error": f"Guardrail Violation: {guard_result.get('violation_type')} - {guard_result.get('reason')}. Rec: {guard_result.get('recommended_action')}",
                "context": context_str
            }

        # 3. Execution (With Context)
        input_prompt = f"Problem: {problem_text}\nContext Highlghts: {context_str}\nPlan: {plan}\nExecute this and return with ---SECTION--- delimiters."
        exec_response = self.executor.run(input_prompt)
        
        # 4. Parsing (Delimited Text - Robust to LaTeX)
        try:
            content_str = exec_response.content
            
            # Default values
            steps = []
            final_answer = "Error"
            confidence = 0.0
            
            # Extract Sections
            if "---STEPS---" in content_str:
                parts = content_str.split("---STEPS---")[1]
                remaining = ""
                
                if "---FINAL_ANSWER---" in parts:
                    steps_part, remaining = parts.split("---FINAL_ANSWER---")
                    
                    # Parse Steps (Split by newline and regex remove numbering "1. ")
                    raw_steps = [line.strip() for line in steps_part.split('\n') if line.strip()]
                    steps = [re.sub(r'^\d+\.\s*', '', s) for s in raw_steps]
                    
                    if "---CONFIDENCE---" in remaining:
                        ans_part, conf_part = remaining.split("---CONFIDENCE---")
                        final_answer = ans_part.strip()
                        try:
                            confidence = float(conf_part.strip())
                        except:
                            confidence = 1.0
                    else:
                        final_answer = remaining.strip()
            
            # Basic validation
            if not steps:
                 # Fallback: maybe they forgot the delimiter?
                 raise ValueError("Could not extract steps with delimiters")

            return {
                "plan": plan,
                "steps": steps,
                "final_answer": final_answer,
                "confidence": confidence,
                "context": context_str
            }

        except Exception as e:
            return {
                "plan": plan,
                "steps": ["Error parsing solver output"],
                "final_answer": "Error",
                "confidence": 0.0,
                "error": f"Solver Parse Error: {str(e)}. Raw: {str(exec_response.content)}"
            }

    def _solve_with_tool(self, problem_text: str) -> Dict[str, Any]:
        """
        Deterministic arithmetic solver using Calculator.
        """
        result = self.calculator.solve_arithmetic(problem_text)
        
        if result["success"]:
            # Format to match normal solver output structure
            return {
                "plan": "Direct computation via SymPy",
                "steps": [
                    f"Parsed expression: {result.get('parsed_expression', 'N/A')}",
                    f"Computed exact value: {result.get('symbolic', 'N/A')}"
                ],
                "final_answer": str(result.get("numerical") or result.get("symbolic")),
                "confidence": 1.0, # Deterministic tools are 100% confident
                "context": "Exact symbolic computation tool used",
                "tool_used": "calculator"
            }
        else:
            # Tool failed - trigger HITL or fallback
            return {
                "plan": "Tool computation failed",
                "steps": [],
                "final_answer": "Error",
                "confidence": 0.0,
                "error": f"Calculator error: {result.get('error')}",
                "tool_used": "calculator"
            }
