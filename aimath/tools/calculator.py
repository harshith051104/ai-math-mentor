
from typing import Dict, Any, Optional, Tuple
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
from agno.agent import Agent
from agno.models.groq import Groq
from aimath.config.settings import Settings

class Calculator:
    """
    Hybrid natural language math calculator.
    Uses regex for common patterns, LLM for complex queries, and SymPy for execution.
    """
    
    def __init__(self):
        # Transformations for robust parsing implies implicit multiplication (e.g. 2x -> 2*x)
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application,)
        )

        # Simple regex patterns (fast path)
        self.simple_patterns = {
            r'square root of\s+(\d+)': lambda m: f'sqrt({m.group(1)})',
            r'sqrt\s*\(?\s*(\d+)\)?': lambda m: f'sqrt({m.group(1)})',
            r'(\d+)\s+squared': lambda m: f'({m.group(1)})**2',
            r'(\d+)\s*\^\s*(\d+)': lambda m: f'({m.group(1)})**{m.group(2)}',
        }
        
        # LLM parser for complex queries
        self.llm_parser = Agent(
            model=Groq(id=Settings.LLM_MODEL),
            description="Math expression translator",
            instructions=[
                "Convert natural language math to SymPy Python syntax.",
                "Output ONLY the expression, no explanation.",
                "Use: sqrt(), sin(), cos(), log(), pi, E, Rational()",
                "Use ** for powers, not ^",
                "If not mathematical, output: ERROR_NOT_MATH"
            ],
            markdown=False
        )
    
    def _try_simple_parse(self, query: str) -> Optional[str]:
        """Try regex patterns first (fast path)."""
        query_lower = query.lower().strip()
        
        for pattern, replacement in self.simple_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                return replacement(match)
        
        return None
    
    def _try_llm_parse(self, query: str) -> Tuple[bool, Optional[str]]:
        """Fallback to LLM for complex queries."""
        try:
            response = self.llm_parser.run(f"Convert: {query}")
            expr = response.content.strip()
            
            if "ERROR_NOT_MATH" in expr:
                return False, None
            
            # Clean artifacts
            expr = expr.replace('```', '').replace('python', '').strip()
            
            return True, expr
            
        except Exception as e:
            return False, None
    
    def parse_query(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Hybrid parsing: Regex -> LLM -> Fail
        
        Returns:
            (success, expression, error_message)
        """
        # 1. Try simple regex (fast)
        simple_expr = self._try_simple_parse(query)
        if simple_expr:
            return True, simple_expr, None
        
        # 2. Try LLM (slower but smarter)
        success, llm_expr = self._try_llm_parse(query)
        if success:
            return True, llm_expr, None
        
        # 3. Failed
        return False, None, "Could not parse mathematical expression"
    
    def compute(self, expression: str) -> Dict[str, Any]:
        """Execute SymPy computation."""
        try:
            # Parse and evaluate
            expr = parse_expr(expression, transformations=self.transformations)
            simplified = sp.simplify(expr)
            
            # Get numerical value
            numerical = None
            try:
                numerical = float(simplified.evalf())
            except:
                pass
            
            return {
                "success": True,
                "symbolic": str(simplified),
                "numerical": numerical,
                "latex": sp.latex(simplified),
                "engine": "sympy"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"SymPy computation failed: {str(e)}"
            }
    
    def solve_arithmetic(self, query: str) -> Dict[str, Any]:
        """
        Main entry point: Natural language -> Result
        
        Examples:
            "square root of 123456" -> 351.363548
            "what is 5 squared?" -> 25
            "calculate 2^10" -> 1024
        """
        # Step 1: Parse natural language
        success, expression, error = self.parse_query(query)
        
        if not success:
            return {
                "success": False,
                "error": error,
                "raw_query": query
            }
        
        # Step 2: Compute with SymPy
        result = self.compute(expression)
        result["parsed_expression"] = expression
        result["original_query"] = query
        
        return result
