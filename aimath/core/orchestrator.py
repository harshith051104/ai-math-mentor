
import uuid
import json
from typing import Optional, Dict, Any
from aimath.config.settings import Settings
from aimath.database.memory import Memory
from aimath.database.vector_store import VectorStore
from aimath.agents.parser_agent import ParserAgent
from aimath.agents.intent_router_agent import IntentRouterAgent
from aimath.agents.solver_agent import SolverAgent
from aimath.agents.verifier_agent import VerifierAgent
from aimath.agents.explainer_agent import ExplainerAgent
from aimath.core.types import PipelineState

class Orchestrator:
    def __init__(self):
        self.memory = Memory()
        self.vector_store = VectorStore()
        
        self.parser = ParserAgent()
        self.router = IntentRouterAgent()
        self.solver = SolverAgent(self.vector_store)
        self.verifier = VerifierAgent()
        self.explainer = ExplainerAgent()

    def start_pipeline(self, text: Optional[str], image_path: Optional[str], audio_path: Optional[str], session_id: Optional[str] = None) -> PipelineState:
        if not session_id:
            session_id = str(uuid.uuid4())
            self.memory.create_session(session_id)
            
        state = PipelineState(
            session_id=session_id,
            raw_input=text,
            input_images=[image_path] if image_path else [],
            input_audio=audio_path
        )
        
        # Step 1: Parse
        parsed = self.parser.run(text, image_path, audio_path)
        state.parsed_data = parsed
        self.memory.log_interaction(session_id, "parser", parsed)
        
        if parsed.get("is_ambiguous"):
            state.needs_hitl = True
            state.hitl_reason = f"Ambiguous Input: {parsed.get('ambiguity_reason')}"
            state.step = "parsed"
            return state

        # Step 2: Route
        category = self.router.route(parsed.get("problem_text", ""))
        state.problem_category = category
        self.memory.log_interaction(session_id, "router", category)
        state.step = "routed"
        
        # Step 3: Solve
        return self.run_solver_phase(state)

    def run_solver_phase(self, state: PipelineState) -> PipelineState:
        # Step 3: Solve
        solution = self.solver.solve(
            state.parsed_data.get("problem_text", ""), 
            problem_type=state.problem_category
        )
        state.solution_steps = solution.get("steps", [])
        state.final_answer = solution.get("final_answer", "")
        state.confidence = solution.get("confidence", 0.0)
        state.solution_plan = solution.get("plan", "")
        # Capture retrieved context for UI display
        state.rag_context = solution.get("context", "No context retrieved")
        state.tool_used = solution.get("tool_used") # Track provenance
        
        self.memory.log_interaction(state.session_id, "solver", solution)
        state.step = "solved"
        
        # Step 4: Verify
        # Reconstruct full solution data for verifier
        solution_data = {
            "steps": state.solution_steps,
            "final_answer": state.final_answer,
            "tool_used": state.tool_used # Critical for trusting calculator
        }
        
        verification = self.verifier.verify(state.parsed_data.get("problem_text", ""), solution_data)
        state.verification_passed = verification["is_correct"]
        state.critique = verification.get("critique", "")
        
        self.memory.log_interaction(state.session_id, "verifier", verification)
        
        # Smart HITL Logic
        solver_conf = state.confidence
        verifier_conf = verification.get("adjusted_confidence", 1.0) # Default to 1.0 if missing
        
        # Default assumption: Needs HITL unless proven otherwise
        state.needs_hitl = True 
        
        if state.tool_used == "calculator":
            # TRUST CALCULATOR: Only trigger HITL if sanity check failed (low verifier confidence)
            if verifier_conf > 0.5:
                state.needs_hitl = False
            else:
                state.hitl_reason = f"Calculator Sanity Check Failed: {state.critique}"
        else:
            # LLM LOGIC: Standard Consensus
            if state.verification_passed and solver_conf > 0.9 and verifier_conf > 0.7:
                 state.needs_hitl = False
            elif state.verification_passed and solver_conf > 0.95:
                 # High solver confidence overrides weak verifier concerns (unless strict fail)
                 state.needs_hitl = False
            else:
                 state.hitl_reason = f"Verification Failed. Solver: {solver_conf}, Verifier: {verifier_conf}, Critique: {state.critique}"
        
        if state.needs_hitl:
            state.step = "verified"
            return state

        # Step 5: Explain
        explanation = self.explainer.explain(
            state.parsed_data.get("problem_text", ""), 
            state.solution_steps, 
            state.final_answer,
            category=state.problem_category
        )
        state.explanation = explanation
        self.memory.log_interaction(state.session_id, "explainer", explanation)
        state.step = "complete"
        
        return state

    def learn_from_success(self, state: PipelineState):
        """
        Self-Learning: Save confirmed correct solutions to Knowledge Base.
        """
        if not state.parsed_data.get("problem_text") or not state.final_answer:
            return
            
        doc_text = f"Problem: {state.parsed_data['problem_text']}\nSolution Plan: {state.solution_plan}\nFinal Answer: {state.final_answer}"
        metadata = {
            "type": "solved_example",
            "category": state.problem_category,
            "confidence": 1.0, # Confirmed by human or consensus
            "source": "user_feedback"
        }
        
        # Use simple ID
        doc_id = f"learn_{state.session_id[:8]}"
        
        try:
            self.vector_store.add_documents([doc_text], [metadata], [doc_id])
            print(f"Learned new pattern: {doc_id}")
        except Exception as e:
            print(f"Failed to learn: {e}")

    def resume_with_feedback(self, state: PipelineState, feedback: Dict[str, Any]) -> PipelineState:
        """
        Resume pipeline after HITL intervention.
        Feedback dict might contain: corrected_text, approved_plan, override_answer, etc.
        """
        
        if "corrected_text" in feedback:
            # Re-parse or just update parsed data
            state.parsed_data["problem_text"] = feedback["corrected_text"]
            state.parsed_data["is_ambiguous"] = False
            state.needs_hitl = False
            
            # Resume directly from Routing
            category = self.router.route(state.parsed_data["problem_text"])
            state.problem_category = category
            return self.run_solver_phase(state)

        if "override_answer" in feedback:
            # User provided the answer directly or corrected it
            state.final_answer = feedback["override_answer"]
            state.needs_hitl = False
            state.verification_passed = True
            state.step = "verified"
            
            # Generate explanation for the user's answer
            explanation = self.explainer.explain(
               state.parsed_data.get("problem_text", ""), 
               ["User provided solution"], 
               state.final_answer,
               category=state.problem_category
            )
            state.explanation = explanation
            state.step = "complete"
            
            # LEARN from this correction
            self.learn_from_success(state)
            
            return state
        
        if "approve" in feedback and feedback["approve"]:
            # Force proceed despite low confidence
            state.needs_hitl = False
            state.verification_passed = True # Override
            
            explanation = self.explainer.explain(
                state.parsed_data.get("problem_text", ""), 
                state.solution_steps, 
                state.final_answer,
                category=state.problem_category
            )
            state.explanation = explanation
            state.step = "complete"
            
            # LEARN from this approval
            self.learn_from_success(state)
            
            return state
        
        return state

    def handle_failure_feedback(self, state: PipelineState) -> PipelineState:
        """
        Production-Grade Feedback Loop:
        1. Log Rich Metadata about the failure.
        2. Trigger Silent Strict Verification.
        """
        # 1. Structured Logging
        feedback_meta = {
            "feedback": "incorrect",
            "problem_type": state.problem_category,
            "solver_confidence": state.confidence,
            "source": "image" if state.input_images else "audio" if state.input_audio else "text"
        }
        print(f"RICH_LOG: {json.dumps(feedback_meta)}")
        # In real app: self.memory.log_feedback_meta(feedback_meta)
        
        # 2. Silent Re-verification (Strict Mode)
        solution_data = {
            "steps": state.solution_steps,
            "final_answer": state.final_answer,
            "tool_used": state.tool_used
        }
        # Call with strict=True
        strict_result = self.verifier.verify(state.parsed_data.get("problem_text", ""), solution_data, strict=True)
        
        # Start tracking that we did a re-check
        state.critique = f"Strict Review: {strict_result.get('critique', 'No issues found')}"
        state.verification_passed = strict_result.get('is_correct', False)
        
        # If strict review failed, we flag it so UI can offer 'Alternative Method'
        if not state.verification_passed:
            state.hitl_reason = "User Rejected + Strict Verify Failed"
            
        return state
