import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
from aimath.core.orchestrator import Orchestrator
from aimath.core.types import PipelineState
from aimath.config.settings import Settings

# Suppress PyTorch CPU warnings
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

st.set_page_config(page_title="Reliable Multimodal Math Mentor", layout="wide")

def save_uploaded_file(uploaded_file, folder="temp"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def render_step_with_latex(step_text):
    """
    Convert inline math to LaTeX rendering with Streamlit
    """
    import re
    # Pattern: $...$ for inline math (assuming standard LaTeX delimiters)
    # Note: Streamlit's st.markdown handles $...$ automatically for KaTeX, 
    # but st.latex provides better block rendering if needed.
    # The user specifically requested this function to handle splitting.
    
    # Split by $...$
    parts = re.split(r'\$([^\$]+)\$', step_text)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular text
            if part.strip():
                st.markdown(part)
        else:
            # Math expression found between $'s
            st.latex(part)

def main():
    st.title("🧮 Reliable Multimodal Math Mentor")
    
    # Check for API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        st.error("🚨 Groq API Key is missing or invalid.")
        st.info("Please open the `.env` file in the project directory (`c:\\Users\\asrit\\OneDrive\\Desktop\\aimath\\.env`) and paste your actual Groq API Key.")
        st.code("GROQ_API_KEY=gsk_...", language="properties")
        st.stop()
    
    # Initialize Orchestrator in Session State
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = None

    with st.sidebar:
        st.header("Input")
        input_type = st.radio("Select Input Type", ["Text", "Image", "Audio"])
        
        text_input = None
        image_path = None
        audio_path = None
        
        if input_type == "Text":
            text_input = st.text_area("Enter Math Problem")
        elif input_type == "Image":
            img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
            if img_file:
                image_path = save_uploaded_file(img_file)
                st.image(image_path, caption="Uploaded Image")
        elif input_type == "Audio":
            output_col, input_col = st.columns([1, 1]) # Optional: Layout tweak if needed, but tabs are better
            
            tab_upload, tab_record = st.tabs(["📂 Upload File", "🎙️ Record Live"])
            
            with tab_upload:
                audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"], key="audio_upload")
                if audio_file:
                    audio_path = save_uploaded_file(audio_file)
                    st.audio(audio_path)
            
            with tab_record:
                # Requires streamlit >= 1.34.0
                recorded_audio = st.audio_input("Record your question")
                if recorded_audio:
                    audio_path = save_uploaded_file(recorded_audio)
                    st.audio(audio_path)

        if st.button("Solve Problem"):
            with st.spinner("Processing..."):
                state = st.session_state.orchestrator.start_pipeline(
                    text=text_input, 
                    image_path=image_path, 
                    audio_path=audio_path
                )
                st.session_state.pipeline_state = state

    # Main Display Area
    state = st.session_state.pipeline_state
    
    if state:
        st.divider()
        st.subheader("Pipeline Status")
        
        # 1. Parsed Data
        with st.expander("1. Input Parsing", expanded=True):
            st.json(state.parsed_data)
            if state.step == "parsed" and state.needs_hitl:
                st.error(f"⚠️ HITL Triggered: {state.hitl_reason}")
                
                with st.form("hitl_parse_form"):
                    corrected_text = st.text_area("Corrected Problem Text", value=state.parsed_data.get("problem_text", ""))
                    if st.form_submit_button("Resume Pipeline"):
                        new_state = st.session_state.orchestrator.resume_with_feedback(state, {"corrected_text": corrected_text})
                        st.session_state.pipeline_state = new_state
                        st.rerun()

        # 2. Routing
        if state.step in ["routed", "solved", "verified", "complete"]:
            with st.expander("2. Intent Routing", expanded=False):
                st.info(f"Category: {state.problem_category}")

        # 3. Solver
        if state.step in ["solved", "verified", "complete"]:
            with st.expander("3. Solver Execution", expanded=False):
                st.write("**Plan:**")
                st.write(state.solution_plan or "N/A - Direct Execution")
                st.write("**Steps:**")
                for i, s in enumerate(state.solution_steps):
                    st.markdown(f"{i+1}. {s}")
                st.write(f"**Final Answer:** {state.final_answer}")
                st.write(f"**Confidence:** {state.confidence}")

        # 4. Verification & HITL
        if state.step in ["verified", "complete", "solved"] and (state.needs_hitl or state.verification_passed):
             with st.expander("4. Verification", expanded=True):
                if state.verification_passed:
                    st.success("✅ Verification Passed")
                else:
                    st.warning(f"⚠️ Verification Issues: {state.critique}")
                
                if state.needs_hitl and state.step == "verified":
                    st.error("Pipeline Paused for Human Review")
                    with st.form("hitl_verify_form"):
                        st.write("Options:")
                        override = st.text_input("Override Final Answer (optional)")
                        approve = st.checkbox("Force Approve (Ignore warnings)")
                        
                        if st.form_submit_button("Submit Decision"):
                            feedback = {}
                            if override:
                                feedback["override_answer"] = override
                            if approve:
                                feedback["approve"] = True
                            
                            new_state = st.session_state.orchestrator.resume_with_feedback(state, feedback)
                            st.session_state.pipeline_state = new_state
                            st.rerun()

        # 5. Explanation
        # 5. Result Display (Enhanced)
        if state.step == "complete":
            st.divider()
            st.success("✅ Solution Complete!")
            
            # Problem Statement
            st.markdown("### 📝 Problem Statement")
            st.info(state.parsed_data.get("problem_text", "N/A"))
            
            explanation = state.explanation if isinstance(state.explanation, dict) else {}
            
            # Concept & Strategy
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📚 Concept")
                st.markdown(f"**Topic:** {explanation.get('concept', 'N/A')}")
                st.markdown(f"**Method:** {explanation.get('strategy', 'N/A')}")
            
            with col2:
                st.markdown("### 📊 Metadata")
                st.metric("Difficulty", explanation.get("difficulty", "Medium"))
                st.metric("JEE Relevance", explanation.get("jee_relevance", "JEE Main"))
                st.metric("Confidence", f"{state.confidence*100:.0f}%")
            
            # Key Insight
            st.markdown("### 💡 Key Insight")
            st.success(explanation.get("key_insight", "No specific insight available."))
            
            # Step-by-Step
            st.markdown("### ✏️ Step-by-Step Solution")
            
            # Tool Badge
            is_tool_used = any("Computed exact value" in s for s in state.solution_steps)
            if is_tool_used:
                st.info("🧮 **Computed using SymPy** (Exact, deterministic calculation)")
            
            for i, step in enumerate(state.solution_steps, 1):
                with st.expander(f"**Step {i}**", expanded=(i==1)):
                    render_step_with_latex(step)
            
            # Final Answer
            st.markdown("---")
            st.markdown("### 🎯 Final Answer")
            
            ans_col1, ans_col2 = st.columns([2, 1])
            with ans_col1:
                # Heuristic: If answer looks like LaTeX, render it
                if "$" in state.final_answer or "\\" in state.final_answer:
                    st.latex(state.final_answer.replace("$", ""))
                else:
                    st.code(state.final_answer, language="text")
            
            with ans_col2:
                 if state.verification_passed:
                    st.success("✅ Verified")
                 else:
                    st.warning("⚠️ Needs Review")
            
            # Learning Points
            st.markdown("---")
            st.markdown("### 🎓 Learning Points")
            for point in explanation.get("learning_points", []):
                st.markdown(f"• {point}")
                
            if explanation.get("common_mistakes"):
                with st.expander("⚠️ Common Mistakes to Avoid"):
                    for mistake in explanation["common_mistakes"]:
                        st.markdown(f"❌ {mistake}")
            
            # Interactive Tabs
            st.markdown("---")
            st.markdown("### 📖 Learn More")
            
            tab1, tab2, tab3 = st.tabs(["📚 Context", "🔍 Agent Trace", "💪 Practice"])
            
            with tab1:
                if state.rag_context:
                    st.markdown("**Retrieved Knowledge:**")
                    st.text(state.rag_context)
                else:
                    st.info("No additional context used.")
            
            with tab2:
                 st.json(state.model_dump())
            
            with tab3:
                st.markdown("**Practice Problems:**")
                st.markdown("1. Find √145800")
                st.markdown("2. Simplify √72 + √128")
                
                if st.button("Generate Similar Problem"):
                    st.info("Generative practice coming soon!")
                    
            # Feedback
            st.markdown("---")
            st.markdown("### 💬 Feedback")
            
            fb_col1, fb_col2, fb_col3 = st.columns(3)
            with fb_col1:
                if st.button("👍 Clear & Helpful", use_container_width=True):
                    st.success("Saved!")
                    st.session_state.orchestrator.learn_from_success(state)
            
            with fb_col2:
                if st.button("🤔 Need More Steps", use_container_width=True):
                    st.info("Feedback recorded.")
            
            with fb_col3:
                if st.button("❌ Incorrect/Unclear", use_container_width=True):
                    new_state = st.session_state.orchestrator.handle_failure_feedback(state)
                    st.session_state.pipeline_state = new_state
                    st.rerun()
             
            # Save Note
            st.markdown("---")
            if st.button("💾 Save to Notes", use_container_width=True):
                note_content = f"# {state.parsed_data.get('problem_text')}\n\n**Answer:** {state.final_answer}\n\n**Insight:** {explanation.get('key_insight')}"
                st.download_button("Download", note_content, "notes.md")

if __name__ == "__main__":
    main()
