
from aimath.agents.explainer_agent import ExplainerAgent

def test_explainer():
    agent = ExplainerAgent()
    
    # Mock data
    problem = "Solve x^2 - 5x + 6 = 0"
    steps = ["Use factorization method", "x^2 - 2x - 3x + 6 = 0", "x(x-2) - 3(x-2) = 0", "(x-2)(x-3) = 0"]
    final_answer = "x = 2, x = 3"
    category = "ALGEBRA"
    
    print("\nRequesting Explanation...")
    result = agent.explain(problem, steps, final_answer, category)
    
    print("\n--- Result Structure ---")
    print(result.keys())
    
    required_keys = ["concept", "strategy", "key_insight", "learning_points", "common_mistakes"]
    missing = [k for k in required_keys if k not in result]
    
    if not missing:
        print("\n✅ JSON Structure Valid")
        print(f"Concept: {result['concept']}")
        print(f"Insight: {result['key_insight']}")
    else:
        print(f"\n❌ Missing Keys: {missing}")

if __name__ == "__main__":
    test_explainer()
