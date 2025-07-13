import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import re
import urllib

from judge import check_lean_proof

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

def get_current_page():
    """Get current page from URL query parameters"""
    query_params = st.query_params
    return query_params.get("page", "landing")

def extract_code (message_content: str):
    final_code=""
    if message_content and "<Result>" in message_content:
        # Extract the final result
        match = re.search(r"<Result>(.*?)</Result>", message_content, re.DOTALL)
        if match:
            final_code = match.group(1).strip()
            final_code = final_code.replace("```lean", "").replace("```", "")

    return final_code


@dataclass
class LeanProblem:
    title: str
    description: str
    specification: str
    difficulty: str = "Medium"

class LeanToolClient:
    """Client for interacting with LeanTool API server"""
    
    def __init__(self, base_url: str = "http://www.codeproofarena.com:8800/v1"):
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
    
    def solve_problem(self, description: str, specification: str, model: str = "sonnet", 
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
                    "role": "user", 
                    "content": f"Please solve the following problem. Description:\n\n{description}\n\nYour solution should satisfy the following Lean 4 specification:\n\n{specification}"
                }
            ],
            "max_tokens": 2000,
            "max_attempts": max_iterations,
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
    

class CodeProofArenaClient:
    """Client for interacting with CodeProofArena API"""
    
    def __init__(self, base_url: str = "http://www.codeproofarena.com:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_problems(self, limit: int = 50) -> List[Dict]:
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
    
    def add_sorry(self, sig:str) ->str:
        if 'sorry' not in sig:
            if not sig.strip().endswith(':='):
                sig += ' := '
            sig += 'sorry'
        return sig

    def _convert_codeproof_format(self, problem: Dict) -> str:
        """Convert CodeProofArena's separate signature format to unified specification"""
        specification_parts = []
        
        # Add function signature if present
        if problem.get("function_signature"):
            specification_parts.append(f"-- Function to implement:")
            specification_parts.append(self.add_sorry(problem["function_signature"]))
            specification_parts.append("")
        
        # Add theorem signature if present  
        if problem.get("theorem_signature"):
            specification_parts.append(f"-- Theorem to prove:")
            specification_parts.append(self.add_sorry(problem["theorem_signature"]))
            specification_parts.append("")
        
        # Add second theorem signature if present
        if problem.get("theorem2_signature"):
            specification_parts.append(f"-- Additional theorem to prove:")
            specification_parts.append(self.add_sorry(problem["theorem2_signature"]))
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


def show_landing_page():
    # Your landing content...
    st.markdown('<h1 class="main-header">Provably-Correct Vibe Coding</h1>', 
                unsafe_allow_html=True)
    st.markdown('<h3 class="main-header">Or "slop that works", depending on your taste in memes...</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
Vibe coding sometimes gets a bad rap, but I believe it reresents a fundamental desire: I have a brilliant idea, and want to turn that into something concrete that people can use. I know what I want to create but don't have the programming expertise. Can my AI assistant help implement my idea into software?

Vibe coding gets the bad rap because it often *doesn't work*.  While the current AI coding assistants can often produce large amounts of plausible-looking code, and perhaps 80% of it is correct; the AI often hallucinates and introduces bugs somewhere in the code. When the AI produces such "slop", our vibe coder would not be able to debug it. Even professional developers may find the debugging effort to be not worth it.

**What if, our AI assistant can produce code that is guaranteed to be correct?** 

*What kind of guarantees?* 

A mathematical proof that the code produces exactly the result that the vibe coder asks for.

*Oh no! Math?? Do I need to understand the proof?*

The good news is that you don't have to understand the proof, because the proof will be machine-checkable. If the proof passes the proof checker program, it is guaranteed to be correct. 

There is a catch: you will need to express the specification in a way that is precise, unambiguous and complete.
             
In this demo, you will be able to pose a coding task as a formal specification
in [Lean](https://lean-lang.org/), a programming language and theorem prover.
The formal specification will take the form of a signature of a function to be implemented, plus theorem statements about the  function.
Then, an AI coding agent will attempt to solve the task by implementing the function (in Lean)
and proving the theorem statements (in Lean).
Finally, you can verify whether the submitted solution is valid by sending it to the Lean proof checker.
If the solution passes the proof checker, you now have runnable code that is guaranteed to be correct.
                
*What kind of AI coding agent?*

The AI agent consists of:
- a base LLM model. This can be any of the commercially available models, including offerings from OpenAI, Anthropic, Google and DeepSeek. 
    You'll need to separately sign up for an API key from one of these providers;                   
- tools made available to the LLMs to facilitate interactions with Lean; 
- a feedback loop, to allow iterative fixing and recursive problem-solving.

*How good is this coding agent? Actually, if the current LLMs hallucinate all the time (as you admitted to earlier),
how do you expect it to output not only correct code, but also correct proof of the code's correctness?*

Excellent question! And that is why if you directly give this task to ChatGPT or Claude.ai, they will likely output something that contains bugs in the code *and* the proof.
However, we have developed *scaffolding* around the LLM, including the tools mentioned above, that allows the LLM to itratively fix the issues in the code and the proof, by interacting with the Lean proof checker.
This is not only able to detect and fix simple syntax errors, but also deeper logical errors in the code implementation that arise from hallucinations. For the latter, we employ *property-based testing* techniques to automatically generate test cases that will be triggered if the code implementation violates the specification.

This is work in progress; I expect we will get stronger and stronger agnents as we get newer
commercial models with better reasoning and coding abilities; and as we get better tooling support, including Lean's hammer tactics that can automatically find proofs for simple proof subgoals.
Meanwhile, try our demo!
"""
    )
    if st.button("🚀 Get Started", type="primary"):
        st.query_params["page"] = "main"
        st.rerun()

    st.markdown("""
If you would like learn more about the open-source technologies behind this demo:
- The tools and feedback loop are provided by [LeanTool](https://github.com/GasStationManager/LeanTool), a library for LLM-Lean interaction.
- The final proof checking of solutions is done by [SafeVerify](https://github.com/GasStationManager/SafeVerify), a utility for safe and robust proof checking in Lean, guarding against potentially adversarial/malicious submissions. 
- Source code for this demo is also [available](https://github.com/GasStationManager/ProvablyCorrectVibeCoding).
                
If you liked the demo and would like to incorporate the technology into your existing AI-assisted development workflow:
[LeanTool](https://github.com/GasStationManager/LeanTool) provides an option to deploy as an [MCP](https://modelcontextprotocol.io/) server,  which allows you to connect to it from
any MCP-supporting coding assistant interface, including Cursor and Claude Code.
                
I encourage you to share your task specifications and solutions, at [Code with Proofs: The Arena](http://www.codeproofarena.com:8000/).
Our demo is able to pull problems from the Arena site to be attempted here.
                
Finally, check out [my blog](https://gasstationmanager.github.io/) where I go into more details on the research efforts to make safe and hallucination-free coding AIs. 
If you are interested in contributing to the project, or just have comments and suggestions, please contact me at GasStationCodeManager@gmail.com
"""
    )
def show_main_app():
    if st.button("← Back to Introduction", type="secondary"):
        st.query_params["page"] = "landing" 
        st.rerun()
    # Main header
    st.markdown('<h1 class="main-header">Provably-Correct Vibe Coding</h1>', 
                unsafe_allow_html=True)
    #st.markdown('<p class="sub-header">Or, "slop that works"</p>', 
    #            unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        # Model selection
        available_models = st.session_state.leantool_client.get_available_models()
        selected_model = st.selectbox(
            "🤖 AI Model",
            available_models,
            help="Choose the AI model for solving problems"
        )        
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
          
        # Solving parameters
        st.subheader("⚙️ Solving Parameters")
        max_iterations = st.slider("Max Iterations", 1, 20, 10)
        #timeout = st.slider("Timeout (seconds)", 30, 600, 300)
        
        # LeanTool info
        #st.subheader("🔧 LeanTool Server")
        #st.info("Using: http://www.codeproofarena.com:8800/v1")
        

    
    # Main content area
    col1, col2 = st.columns([0.4, 0.6])
    
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
                problem_description=st.text_area("Description",value=problem.description,height=300)
                
                lean_specification = st.text_area(
                    "Lean Specification",
                    value=problem.specification,
                    height=300,
                    help="Edit the Lean specification if needed"
                )

        
        elif problem_source == "Custom Input":
            problem_description = st.text_area(
                "Problem Description",
                height=300
            )
            lean_specification = st.text_area(
                "Enter your Lean specification",
                placeholder="-- Enter your Lean specification here...\ndef my_function := sorry\ntheorem my_theorem : my_property := by sorry",
                height=300
            )
        
        elif problem_source == "CodeProofArena":
            if "arena_problems" not in st.session_state:
              problems = st.session_state.codeproof_client.get_problems()
              if problems:
                st.success(f"✅ Loaded {len(problems)} problems")
                st.session_state.arena_problems = problems
              else:
                st.warning("No problems loaded (check connection)")
            if "arena_problems" in st.session_state and st.session_state.arena_problems:
                problem_titles = [f"{p.get('title', 'Untitled')} (ID: {p.get('id', 'N/A')})" 
                                for p in st.session_state.arena_problems]
                arena_problem = st.selectbox("Choose from CodeProofArena", problem_titles)
                
                if arena_problem:
                    # Extract problem specification from selected arena problem
                    problem_idx = problem_titles.index(arena_problem)
                    selected_arena_problem = st.session_state.arena_problems[problem_idx]
                    
                    # Show problem details
                    
                    st.markdown(f"**Difficulty:** {selected_arena_problem.get('difficulty', 'Unknown')}")
                    problem_description=st.text_area("Description:",value= selected_arena_problem.get('description', 'No description'), height=300)
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
                            st.code(original['function_signature'], language='lean')
                        if original.get('theorem_signature'):
                            st.code(original['theorem_signature'], language='lean')
                        if original.get('theorem2_signature'):
                            st.code(original['theorem2_signature'], language='lean')




        
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
                    problem_description,
                    lean_specification, 
                    selected_model, 
                    st.session_state.api_key,
                    max_iterations, 
                )
                
                if result.get("success", False):
                    st.session_state.current_solution = result.get("solution", "")
                    st.write("✅ Solution generated!")
                    status.update(label="✅ Problem solved!", state="complete")
                else:
                    st.session_state.current_solution=result.get('error', 'Unknown error')
                    st.write(f"❌ Error: {st.session_state.current_solution}")
                    status.update(label="❌ Solving failed", state="error")

                st.session_state.last_verification = None
                st.session_state.solving_in_progress = False
                time.sleep(1)  # Brief pause before rerun
                st.rerun()
    
    with col2:
        st.header("✅ Generated Solution")
        
        if st.session_state.current_solution:
            # Display the solution
            st.markdown(st.session_state.current_solution)
            current_code=extract_code(st.session_state.current_solution)
            if current_code:
              st.subheader("Code")
              st.code(current_code, language="lean")
            
            # Verification section
            st.subheader("🔍 Verification")
            
            col_verify, col_playground, col_download = st.columns([1, 1, 1])
            
            with col_verify:
                if st.button("Verify Solution", type="secondary", disabled=(not current_code.strip())):

                        with st.spinner("Verifying solution..."):
                            verification_result = check_lean_proof(
                                lean_specification,
                                current_code,
                            )
                            st.session_state.last_verification = verification_result
                            st.rerun()
            with col_playground:
                st.link_button(
                    "Send to live.lean-lang.org Playground",
                    'https://live.lean-lang.org/#code='+urllib.parse.quote(current_code)
                )
            
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
                if verification.get("is_correct", False):
                    st.markdown(
                        '<div class="success-box">✅ <strong>Verification Passed!</strong> </div>',
                        unsafe_allow_html=True
                    )
                    # Show verification response if available
                    if verification.get("feedback"):
                        with st.expander("View Verification Details"):
                            st.text(verification["feedback"])
                else:
                    
                    response = verification.get("feedback", "")
                    st.markdown(
                        f'<div class="error-box">❌ <strong>Verification Issues Found:</strong></div>',
                        unsafe_allow_html=True
                    )
                    if response:
                        st.text_area("Verification Feedback:", value=response, height=100)

            
            
            # CodeProofArena submission (placeholder for future implementation)
            if problem_source == "CodeProofArena" and "arena_problems" in st.session_state:
                st.subheader("📤 Submit to Arena")
                st.info("🚧 **Coming Soon:** Direct submission to [CodeProofArena](http://www.codeproofarena.com:8000/). Meanwhile, feel free to paste the solution over there!")
        
        else:
            st.markdown(
                '<div class="info-box">💡 <strong>Ready to solve!</strong><br>'
                'Enter a Lean specification on the left and click "Solve Problem" to get started.</div>',
                unsafe_allow_html=True
            )
            

def main():
    initialize_session_state()
    # Route based on URL query parameter
    current_page = get_current_page()
    
    if current_page == "landing":
        show_landing_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
