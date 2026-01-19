
from aimath.database.vector_store import VectorStore

def seed():
    print("🌱 Seeding ChromaDB with JEE Math Knowledge...")
    store = VectorStore()
    
    # Curated Knowledge Base (15 Essential Docs)
    documents = [
        # Algebra - Quadratics
        "Topic: Quadratic Equations. Formula: For ax^2 + bx + c = 0, roots are x = (-b ± sqrt(b^2 - 4ac)) / 2a. Nature of roots: D > 0 distinct real, D = 0 equal real, D < 0 complex. Sum of roots = -b/a, Product = c/a.",
        "Topic: Vieta's Formulas. For cubic ax^3 + bx^2 + cx + d = 0: Sum roots = -b/a, Sum product pairs = c/a, Product roots = -d/a.",
        
        # Algebra - Sequences
        "Topic: Arithmetic Progression (AP). Nth term: an = a + (n-1)d. Sum: Sn = n/2 * (2a + (n-1)d).",
        "Topic: Geometric Progression (GP). Nth term: an = a * r^(n-1). Sum: Sn = a(r^n - 1)/(r-1). Sum infinite: S_inf = a/(1-r) for |r| < 1.",
        
        # Trigonometry
        "Topic: Trigonometric Identities. sin^2 + cos^2 = 1. 1 + tan^2 = sec^2. 1 + cot^2 = csc^2. sin(2A) = 2sinAcosA. cos(2A) = cos^2A - sin^2A.",
        "Topic: General Solutions of Trig Equations. If sin(x) = sin(alpha), x = n*pi + (-1)^n * alpha. If cos(x) = cos(alpha), x = 2n*pi ± alpha.",
        
        # Calculus - Limits
        "Topic: Standard Limits. lim(x->0) sin(x)/x = 1. lim(x->0) (e^x - 1)/x = 1. lim(x->0) ln(1+x)/x = 1.",
        
        # Calculus - Derivatives
        "Topic: derivative rules. d/dx(sin x) = cos x. d/dx(cos x) = -sin x. d/dx(ln x) = 1/x. Chain Rule: d/dx f(g(x)) = f'(g(x)) * g'(x).",
        "Topic: Maxima and Minima. For y=f(x), find dy/dx = 0 for critical points. Check d^2y/dx^2: if > 0 minima, if < 0 maxima.",
        
        # Calculus - Integration
        "Topic: Standard Integrals. int x^n dx = x^(n+1)/(n+1). int 1/x dx = ln|x|. int e^x dx = e^x.",
        
        # Coordinate Geometry
        "Topic: Straight Lines. Slope m = (y2-y1)/(x2-x1). Point-Slope: y - y1 = m(x - x1). Parallel lines have same slope m1=m2. Perpendicular m1*m2 = -1.",
        "Topic: Circles. Standard eq: (x-h)^2 + (y-k)^2 = r^2. General eq: x^2 + y^2 + 2gx + 2fy + c = 0, Center (-g, -f), Radius sqrt(g^2 + f^2 - c).",
        
        # Vectors
        "Topic: Vector Product. Dot Product a.b = |a||b|cos(theta). Cross Product a x b = |a||b|sin(theta) n_cap. Projection of a on b = (a.b)/|b|.",
        
        # Probability
        "Topic: Probability Basics. P(A) = Favorable/Total. P(A or B) = P(A) + P(B) - P(A and B). Conditional P(A|B) = P(A and B) / P(B).",
        
        # Common Pitfalls
        "Common Pitfall: Division by Zero. Never cancel variables without checking if they are zero. Example: x/x = 1 only if x != 0.",
        "Common Pitfall: Squaring Equations. Squaring both sides can introduce extraneous roots. Always verify answers in original equation.",
        "Common Pitfall: Domain of Log. For log_b(a), a > 0, b > 0, b != 1."
    ]
    
    metadatas = [{"type": "knowledge", "source": "jee_curated"} for _ in documents]
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    store.add_documents(documents, metadatas, ids)
    print(f"✅ Successfully seeded {len(documents)} documents.")

if __name__ == "__main__":
    seed()
