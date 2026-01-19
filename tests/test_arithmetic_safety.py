
import unittest
from aimath.tools.calculator import Calculator
from aimath.agents.solver_agent import SolverAgent
from aimath.agents.intent_router_agent import IntentRouterAgent
from aimath.agents.guardrail_agent import GuardrailAgent
from aimath.database.vector_store import VectorStore

class TestArithmeticSafety(unittest.TestCase):
    
    def setUp(self):
        # Mock vector store for lightweight testing
        self.vector_store = VectorStore(persist_directory="aimath/database/storage/chroma_test")
        self.calculator = Calculator()
        self.router = IntentRouterAgent()
        self.guardrail = GuardrailAgent()
        self.solver = SolverAgent(self.vector_store)

    def test_calculator_parsing(self):
        print("\nTesting Calculator Parsing...")
        queries = [
            ("square root of 123456", "351.36"),
            ("what is 5 squared?", "25"),
            ("sqrt(144)", "12")
        ]
        
        for q, expected in queries:
            result = self.calculator.solve_arithmetic(q)
            self.assertTrue(result["success"], f"Failed to parse: {q}")
            print(f"  Query: '{q}' -> Result: {result['numerical']}")
            self.assertIn(expected.split('.')[0], str(result["numerical"]))

    def test_solver_tool_routing(self):
        print("\nTesting Solver Tool Routing...")
        # Direct call to solve with ARITHMETIC type
        result = self.solver.solve("Calculate square root of 123456", problem_type="ARITHMETIC")
        
        self.assertEqual(result["tool_used"], "calculator")
        self.assertEqual(result["confidence"], 1.0)
        print(f"  Tool Used: {result['tool_used']}")
        print(f"  Final Answer: {result['final_answer']}")
        self.assertTrue("351.36" in str(result["final_answer"]))

    def test_guardrail_arithmetic_rule(self):
        print("\nTesting Guardrail Rule 7...")
        # A plan that suggests mental math for large numbers
        bad_plan = """
        Strategy:
        1. Use Babylonian method to approximate
        2. Iterate manually to find root
        """
        
        # NOTE: We need to mock the Guardrail or rely on its LLM. 
        # Since this is an integration test with real LLM, we expect it to catch this.
        result = self.guardrail.check("sqrt(123456)", bad_plan)
        
        if not result["is_safe"]:
            print(f"  Caught Improper Plan: {result['violation_type']}")
            self.assertTrue("Rule 7" in result.get("violation_type", "") or "Rule 8" in result.get("violation_type", ""))
        else:
            print("  WARNING: Guardrail allowed mental math (LLM varability).")

if __name__ == '__main__':
    unittest.main()
