import json
import re
from typing import Dict, Any, Optional
from agno.agent import Agent
from agno.models.groq import Groq
from aimath.config.settings import Settings

# Note: In a real production setup, OCR/ASR models might be loaded globally or in a separate service to avoid reloading.
try:
    import easyocr
    reader = easyocr.Reader(['en'])
except ImportError:
    reader = None

try:
    from faster_whisper import WhisperModel
    # "tiny" for speed in this demo, "large-v3" for production
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
except ImportError:
    whisper_model = None

class ParserAgent:
    """
    Cleaner & Structurer Agent.
    Responsibilities:
    1. Accept Multimodal Input (Text, Image Path, Audio Path)
    2. Extract Raw Text (OCR/ASR) with Deterministic Guards
    3. LLM Cleaning & Structuring to JSON
    4. Ambiguity Detection
    """
    
    def __init__(self):
        self.agent = Agent(
            model=Groq(id=Settings.LLM_MODEL),
            description="You are a Data Extraction Engine.",
            instructions=[
                "Input: User text (raw).",
                "Task: Convert input into STRICT JSON.",
                "Rules:",
                "1. Extract the core math problem into 'problem_text'.",
                "2. Identify input type ('text', 'latex').",
                "3. Detect ambiguity. If inputs are conflicting or gibberish, set 'is_ambiguous': true.",
                "4. DO NOT solve the problem.",
                "5. DO NOT normalize or paraphrase the problem text.",
                "Output STRICT JSON format with keys: 'problem_text', 'input_type', 'is_ambiguous', 'ambiguity_reason'."
            ],
            markdown=False  # Reduce JSON artifacts
        )

    def looks_corrupted(self, text: str) -> bool:
        """
        Deterministic check for corrupted OCR/ASR output.
        Returns True if text looks like garbage or severe hallucination.
        """
        if not text or len(text) < 3:
            return True
            
        # Common OCR artifact patterns
        corruption_markers = ["&", "~", "Xt", "X1", "?=", "--", "|", "ERROR", "sec8", "tane", "α1", "β1"]
        if any(marker in text for marker in corruption_markers):
            return True
            
        # High noise ratio check (e.g., if >50% chars are non-alphanumeric and non-space)
        clean_chars = sum(c.isalnum() or c.isspace() for c in text)
        if clean_chars / len(text) < 0.5:
            return True
            
        return False

    def process_image(self, image_path: str) -> str:
        if not reader:
            return "ERROR_OCR_MISSING"
        try:
            results = reader.readtext(image_path, detail=0)
            text = " ".join(results)
            return text
        except Exception as e:
            return "ERROR_OCR_FAILED"

    def process_audio(self, audio_path: str) -> str:
        if not whisper_model:
            return "ERROR_ASR_MISSING"
        try:
            segments, info = whisper_model.transcribe(audio_path, beam_size=5)
            text = " ".join([segment.text for segment in segments])
            return text
        except Exception as e:
            return "ERROR_ASR_FAILED"

    def run(self, text: Optional[str] = None, image_path: Optional[str] = None, audio_path: Optional[str] = None) -> Dict[str, Any]:
        raw_inputs = []
        input_sources = []
        
        # 1. Extraction & Deterministic Corruption Check
        if image_path:
            ocr_text = self.process_image(image_path)
            
            # Immediate Corruption Check
            if self.looks_corrupted(ocr_text):
                return {
                    "problem_text": "",
                    "input_type": "image",
                    "is_ambiguous": True,
                    "ambiguity_reason": "OCR output is corrupted or unreadable",
                    "raw_input_combined": f"OCR Output: {ocr_text}",
                    "sources": ["image"]
                }
                
            raw_inputs.append(f"OCR Output: {ocr_text}")
            input_sources.append("image")
            
        if audio_path:
            asr_text = self.process_audio(audio_path)
            
            # Immediate Corruption Check for Audio too
            if self.looks_corrupted(asr_text):
                 return {
                    "problem_text": "",
                    "input_type": "audio",
                    "is_ambiguous": True,
                    "ambiguity_reason": "Audio transcript is corrupted or unintelligible",
                    "raw_input_combined": f"Audio Output: {asr_text}",
                    "sources": ["audio"]
                }
            
            raw_inputs.append(f"Audio Transcript: {asr_text}")
            input_sources.append("audio")
            
        if text:
            raw_inputs.append(f"User Text: {text}")
            input_sources.append("text")
        
        if not raw_inputs:
             return {
                "problem_text": "",
                "input_type": "none",
                "is_ambiguous": True,
                "ambiguity_reason": "No valid input provided",
                "raw_input_combined": "",
                "sources": []
            }

        full_input_str = "\n".join(raw_inputs)
        
        # 2. LLM Structuring
        response = self.agent.run(f"Process this raw input into structured JSON: \n{full_input_str}")
        
        # 3. Safe Parsing & Schema Enforcement
        try:
            content_str = response.content
            # Cleanup markdown if model ignores markdown=False
            match = re.search(r"```json\s*(.*?)```", content_str, re.DOTALL)
            if match:
                content_str = match.group(1).strip()
            else:
                start = content_str.find('{')
                end = content_str.rfind('}') + 1
                if start != -1 and end != 0:
                    content_str = content_str[start:end]
            
            data = json.loads(content_str)
            
            # MANDATORY SCHEMA CHECK
            required_keys = {"problem_text", "input_type", "is_ambiguous", "ambiguity_reason"}
            if not required_keys.issubset(data.keys()):
                raise ValueError(f"Missing required JSON keys. Found: {data.keys()}")

            # 4. Ambiguity Contract Enforcement
            # Rule: If reason exists -> is_ambiguous MUST be True
            if data.get("ambiguity_reason"):
                data["is_ambiguous"] = True
            elif not data.get("ambiguity_reason") and data.get("is_ambiguous"):
                # If ambiguous but no reason, provide default
                data["ambiguity_reason"] = "Ambiguous input detected (unspecified by parser)"
            else:
                # Not ambiguous -> Reason must be None
                data["ambiguity_reason"] = None

            data['raw_input_combined'] = full_input_str
            data['sources'] = input_sources
            return data

        except Exception as e:
            return {
                "problem_text": full_input_str,
                "input_type": input_sources,
                "is_ambiguous": True,
                "ambiguity_reason": f"Parser Error: {str(e)}",
                "raw_response": str(response.content)
            }
