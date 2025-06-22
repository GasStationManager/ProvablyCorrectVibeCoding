import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Provably-Correct Vibe Coding",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #666;
    margin-bottom: 2rem;
}
.success-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.error-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}
.info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
}
</style>
""", unsafe_allow_html=True)

@dataclass
class LeanProblem:
    title: str
    description: str
    specification: str
    difficulty: str = "Medium"

class LeanToolClient:
    """Client for interacting with LeanTool API server"""
    
    def __init__(self, base_url: str = "http://codeproofarena.com:8800/v1"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from LeanTool"""
        # Since we're using OpenAI-compatible API, return common model names
        return [
            "sonnet", 
            "opus",
            "deepseek",
            "r1",
            "o3",
            "o4-mini",
            "gemini-pro",
        ]
    
    def solve_problem(self, specification: str, model: str = "gpt-4", 
                     api_key: str = "", max_iterations: int = 10, timeout: int = 300) -> Dict:
        """Send problem to LeanTool for solving using OpenAI-compatible API"""
        
        # Prepare headers for OpenAI-compatible API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Format as OpenAI chat completion request
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": f"You are an expert Lean 4 theorem prover. Solve the following specification by implementing the required functions and proving the theorems. You have {max_iterations} iterations to get it right."
                },
                {
                    "role": "user", 
                    "content": f"Please solve this Lean 4 specification:\n\n{specification}"
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout + 30,
                stream=False
            )
            
            if response.status_code == 200:
                result = response.json()
                solution = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "solution": solution,
                    "model_used": model
                }
            else:
                return {
                    "success": False,
                    "error": f"API returned status {response.status_code}: {response.text}"
                }
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": f"Network error: {str(e)}"}
    
    def verify_solution(self, code: str, api_key: str = "") -> Dict:
        """Verify a Lean solution by asking the model to check it"""
        headers = {
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4",  # Use a reliable model for verification
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Lean 4 expert. Check if the provided code is syntactically correct and if the proofs are complete. Respond with 'VERIFIED' if correct, or explain the issues."
                },
                {
                    "role": "user",
                    "content": f"Please verify this Lean 4 code:\n\n{code}"
                }
            ],
            "max_tokens": 1000,
            "temperature": 0
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                verification_response = result["choices"][0]["message"]["content"]
                verified = "VERIFIED" in verification_response.upper()
                return {
                    "verified": verified,
                    "response": verification_response
                }
            else:
                return {
                    "verified": False,
                    "error": f"Verification failed with status {response.status_code}"
                }
        except Exception as e:
            return {"verified": False, "error": f"Verification error: {str(e)}"}

class CodeProofArenaClient:
    """Client for interacting with CodeProofArena API"""
    
    def __init__(self, base_url: str = "http://www.codeproofarena.com:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_problems(self, limit: int = 10) -> List[Dict]:
        """Fetch problems from CodeProofArena and convert to unified format"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/challenges",
                params={"limit": limit},
                timeout=10
            )
            if response.status_code == 200:
                problems = response.json()
                # Convert CodeProofArena format to unified format
                unified_problems = []
                for problem in problems:
                    unified_spec = self._convert_codeproof_format(problem)
                    unified_problems.append({
                        "id": problem.get("id"),
                        "title": problem.get("title", "Untitled"),
                        "description": problem.get("description", ""),
                        "specification": unified_spec,
                        "difficulty": problem.get("difficulty", "Medium"),
                        "original_format": problem  # Keep original for reference
                    })
                return unified_problems
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to fetch problems: {e}")
            return []
    
    def _convert_codeproof_format(self, problem: Dict) -> str:
        """Convert CodeProofArena's separate signature format to unified specification"""
        specification_parts = []
        
        # Add function signature if present
        if problem.get("function_signature"):
            specification_parts.append(f"-- Function to implement:")
            specification_parts.append(problem["function_signature"])
            specification_parts.append("")
        
        # Add theorem signature if present  
        if problem.get("theorem_signature"):
            specification_parts.append(f"-- Theorem to prove:")
            specification_parts.append(problem["theorem_signature"])
            specification_parts.append("")
        
        # Add second theorem signature if present
        if problem.get("theorem2_signature"):
            specification_parts.append(f"-- Additional theorem to prove:")
            specification_parts.append(problem["theorem2_signature"])
            specification_parts.append("")
        
        # If no structured signatures, fall back to description or generic format
        if not specification_parts:
            title = problem.get("title", "Problem")
            description = problem.get("description", "")
            specification_parts = [
                f"-- {title}",
                f"-- {description}" if description else "",
                "-- Please implement the required functions and prove the theorems",
                "",
                "def solution := sorry",
                "theorem main_theorem : sorry := by sorry"
            ]
        
        return "\n".join(specification_parts)
    
    def submit_solution(self, challenge_id: str, solution: str) -> Dict:
        """Submit solution to CodeProofArena"""
        try:
            payload = {
                "challenge_id": challenge_id,
                "solution": solution,
                "language": "lean4"
            }
            response = self.session.post(
                f"{self.base_url}/api/submissions",
                json=payload,
                timeout=30
            )
            return response.json() if response.status_code == 201 else {"success": False}
        except Exception as e:
            logger.error(f"Failed to submit solution: {e}")
            return {"success": False, "error": str(e)}

# Sample problems for testing
SAMPLE_PROBLEMS = {
    "Sorting Correctness": LeanProblem(
        title="Sorting Correctness",
        description="Implement and prove correctness of a sorting function for natural numbers.",
        specification="""-- Implement a sorting function and prove it works correctly
def my_sort (l : List Nat) : List Nat := sorry

theorem sort_correct (l : List Nat) : 
  (my_sort l).Sorted (· ≤ ·) ∧ 
  l.toMultiset = (my_sort l).toMultiset := by sorry""",
        difficulty="Medium"
    ),
    
    "Binary Search": LeanProblem(
        title="Binary Search",
        description="Implement binary search with a correctness proof.",
        specification="""-- Implement binary search with correctness proof
def binary_search (arr : Array Nat) (target : Nat) : Option Nat := sorry

theorem binary_search_correct (arr : Array Nat) (target : Nat) 
  (h : arr.toList.Sorted (· ≤ ·)) :
  match binary_search arr target with
  | some i => arr[i]? = some target
  | none => target ∉ arr.toList := by sorry""",
        difficulty="Hard"
    ),
    
    "Addition Commutativity": LeanProblem(
        title="Addition Commutativity",
        description="Prove that addition is commutative for natural numbers.",
        specification="""-- Prove addition is commutative
theorem add_comm_proof (a b : Nat) : a + b = b + a := by sorry""",
        difficulty="Easy"
    ),
    
    "List Reverse": LeanProblem(
        title="List Reverse",
        description="Implement list reversal and prove that reversing twice gives the original list.",
        specification="""-- Implement list reverse and prove double reverse property
def my_reverse (l : List α) : List α := sorry

theorem reverse_reverse (l : List α) : 
  my_reverse (my_reverse l) = l := by sorry""",
        difficulty="Medium"
    )
}

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "leantool_client" not in st.session_state:
        st.session_state.leantool_client = LeanToolClient()  # Initialize with hardcoded URL
    if "codeproof_client" not in st.session_state:
        st.session_state.codeproof_client = CodeProofArenaClient()
    if "current_solution" not in st.session_state:
        st.session_state.current_solution = ""
    if "solving_in_progress" not in st.session_state:
        st.session_state.solving_in_progress = False
    if "last_verification" not in st.session_state:
        st.session_state.last_verification = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

def main():
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🧮 Provably-Correct Vibe Coding</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate formally verified code with AI assistance</p>', 
                unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        st.subheader("🔑 API Key")
        api_key = st.text_input(
            "Enter your API key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your API key for OpenAI, Anthropic, Google, or OpenRouter"
        )
        st.session_state.api_key = api_key
        
        if api_key:
            st.success("✅ API key configured")
        else:
            st.warning("⚠️ Please enter your API key to use the service")
        
        # Model selection
        available_models = st.session_state.leantool_client.get_available_models()
        selected_model = st.selectbox(
            "🤖 AI Model",
            available_models,
            help="Choose the AI model for solving problems"
        )
        
        # Solving parameters
        st.subheader("⚙️ Solving Parameters")
        max_iterations = st.slider("Max Iterations", 1, 20, 10)
        timeout = st.slider("Timeout (seconds)", 30, 600, 300)
        
        # LeanTool info
        st.subheader("🔧 LeanTool Server")
        st.info("Using: http://codeproofarena.com:8800/v1")
        
        # CodeProofArena integration
        st.subheader("🏟️ CodeProofArena")
        if st.button("Load Problems from Arena"):
            with st.spinner("Loading problems..."):
                problems = st.session_state.codeproof_client.get_problems()
                if problems:
                    st.success(f"✅ Loaded {len(problems)} problems")
                    st.session_state.arena_problems = problems
                else:
                    st.warning("No problems loaded (check connection)")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Problem Specification")
        
        # Problem source selection
        problem_source = st.radio(
            "Problem Source",
            ["Sample Problems", "Custom Input", "CodeProofArena"],
            horizontal=True
        )
        
        if problem_source == "Sample Problems":
            selected_problem = st.selectbox(
                "Choose a sample problem",
                list(SAMPLE_PROBLEMS.keys())
            )
            
            if selected_problem:
                problem = SAMPLE_PROBLEMS[selected_problem]
                st.markdown(f"**Difficulty:** {problem.difficulty}")
                st.markdown(f"**Description:** {problem.description}")
                
                lean_specification = st.text_area(
                    "Lean Specification",
                    value=problem.specification,
                    height=300,
                    help="Edit the Lean specification if needed"
                )
        
        elif problem_source == "Custom Input":
            lean_specification = st.text_area(
                "Enter your Lean specification",
                placeholder="-- Enter your Lean specification here...\ndef my_function := sorry\ntheorem my_theorem : my_property := by sorry",
                height=300
            )
        
        elif problem_source == "CodeProofArena":
            if "arena_problems" in st.session_state and st.session_state.arena_problems:
                problem_titles = [f"{p.get('title', 'Untitled')} (ID: {p.get('id', 'N/A')})" 
                                for p in st.session_state.arena_problems]
                arena_problem = st.selectbox("Choose from CodeProofArena", problem_titles)
                
                if arena_problem:
                    # Extract problem specification from selected arena problem
                    problem_idx = problem_titles.index(arena_problem)
                    selected_arena_problem = st.session_state.arena_problems[problem_idx]
                    
                    # Show problem details
                    st.markdown(f"**Description:** {selected_arena_problem.get('description', 'No description')}")
                    st.markdown(f"**Difficulty:** {selected_arena_problem.get('difficulty', 'Unknown')}")
                    
                    # Show the unified specification (converted from CodeProofArena format)
                    lean_specification = st.text_area(
                        "Lean Specification (converted from Arena format)",
                        value=selected_arena_problem.get('specification', ''),
                        height=300,
                        help="This specification was automatically converted from CodeProofArena's format"
                    )
                    
                    # Show original format in expander for reference
                    with st.expander("View Original CodeProofArena Format"):
                        original = selected_arena_problem.get('original_format', {})
                        if original.get('function_signature'):
                            st.code(original['function_signature'], language='lean', help="Function Signature")
                        if original.get('theorem_signature'):
                            st.code(original['theorem_signature'], language='lean', help="Theorem Signature")
                        if original.get('theorem2_signature'):
                            st.code(original['theorem2_signature'], language='lean', help="Additional Theorem")
            else:
                st.info("Load problems from CodeProofArena first (use sidebar)")
                lean_specification = ""
        
        # Solve button
        solve_disabled = (not lean_specification.strip() or 
                         st.session_state.solving_in_progress or 
                         not st.session_state.api_key.strip())
        
        if st.button("🚀 Solve Problem", 
                    type="primary", 
                    disabled=solve_disabled,
                    help="Send the problem to AI for solving"):
            if st.session_state.api_key.strip() and lean_specification.strip():
                st.session_state.solving_in_progress = True
                st.rerun()
        
        # Show solving status
        if st.session_state.solving_in_progress:
            with st.status("🤖 AI is working on your problem...", expanded=True) as status:
                st.write("Sending specification to LeanTool...")
                
                # Call LeanTool API
                result = st.session_state.leantool_client.solve_problem(
                    lean_specification, 
                    selected_model, 
                    st.session_state.api_key,
                    max_iterations, 
                    timeout
                )
                
                if result.get("success", False):
                    st.session_state.current_solution = result.get("solution", "")
                    st.write("✅ Solution generated!")
                    status.update(label="✅ Problem solved!", state="complete")
                else:
                    st.write(f"❌ Error: {result.get('error', 'Unknown error')}")
                    status.update(label="❌ Solving failed", state="error")
                
                st.session_state.solving_in_progress = False
                time.sleep(1)  # Brief pause before rerun
                st.rerun()
    
    with col2:
        st.header("✅ Generated Solution")
        
        if st.session_state.current_solution:
            # Display the solution
            st.code(st.session_state.current_solution, language="lean")
            
            # Verification section
            st.subheader("🔍 Verification")
            
            col_verify, col_download = st.columns([1, 1])
            
            with col_verify:
                if st.button("Verify Solution", type="secondary"):
                    if st.session_state.api_key.strip():
                        with st.spinner("Verifying solution..."):
                            verification_result = st.session_state.leantool_client.verify_solution(
                                st.session_state.current_solution,
                                st.session_state.api_key
                            )
                            st.session_state.last_verification = verification_result
                            st.rerun()
                    else:
                        st.error("Please enter your API key first")
            
            with col_download:
                st.download_button(
                    "Download Solution",
                    data=st.session_state.current_solution,
                    file_name="lean_solution.lean",
                    mime="text/plain"
                )
            
            # Show verification results
            if st.session_state.last_verification:
                verification = st.session_state.last_verification
                if verification.get("verified", False):
                    st.markdown(
                        '<div class="success-box">✅ <strong>Verification Passed!</strong> The AI believes the solution is correct.</div>',
                        unsafe_allow_html=True
                    )
                    # Show verification response if available
                    if verification.get("response"):
                        with st.expander("View Verification Details"):
                            st.text(verification["response"])
                else:
                    error_msg = verification.get("error", "Verification failed")
                    response = verification.get("response", "")
                    st.markdown(
                        f'<div class="error-box">❌ <strong>Verification Issues Found:</strong></div>',
                        unsafe_allow_html=True
                    )
                    if response:
                        st.text_area("Verification Feedback:", value=response, height=100)
                    if error_msg and error_msg != "Verification failed":
                        st.error(f"Error: {error_msg}")
            
            # Note about verification
            st.info("💡 **Note:** Verification uses AI analysis. For production use, integrate with actual Lean checker or SafeVerify.")
            
            # CodeProofArena submission (placeholder for future implementation)
            if problem_source == "CodeProofArena" and "arena_problems" in st.session_state:
                st.subheader("📤 Submit to Arena")
                st.info("🚧 **Coming Soon:** Direct submission to CodeProofArena will require username/password authentication.")
        
        else:
            st.markdown(
                '<div class="info-box">💡 <strong>Ready to solve!</strong><br>'
                'Enter a Lean specification on the left and click "Solve Problem" to get started.</div>',
                unsafe_allow_html=True
            )
            
            # Show example
            st.subheader("Example Problem")
            st.code('''-- Example: Prove addition is commutative
theorem add_comm_proof (a b : Nat) : a + b = b + a := by sorry''', language="lean")

if __name__ == "__main__":
    main()