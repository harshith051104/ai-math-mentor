from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PipelineState(BaseModel):
    session_id: str
    step: str = "init" # init, parsed, routed, solved, verified, explained, complete
    
    # Data accumulating through the pipeline
    raw_input: Optional[str] = None
    input_images: List[str] = []
    input_audio: Optional[str] = None
    
    parsed_data: Dict[str, Any] = {}
    problem_category: Optional[str] = None
    solution_plan: Optional[str] = None
    solution_steps: List[str] = []
    final_answer: Optional[str] = None
    confidence: float = 0.0
    tool_used: Optional[str] = None
    
    # Context
    rag_context: Optional[str] = None
    
    verification_passed: bool = False
    critique: Optional[str] = None
    
    explanation: Optional[Any] = None
    
    # Flags
    needs_hitl: bool = False
    hitl_reason: Optional[str] = None
