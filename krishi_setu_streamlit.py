import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, OpenWeatherMapAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool

# --- PAGE CONFIG ---
st.set_page_config(page_title="Krishi Setu - AI Farmer Advisor", page_icon="🌾", layout="wide")

# --- INITIALIZATION ---
# Load .env from the same directory or generativeAI folder
load_dotenv(dotenv_path="generativeAI/.env")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")

# Custom Styling
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .main {
        background-color: #f5f7f9;
    }
    .stSidebar {
        background-color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# --- MEMORY PERSISTENCE ---
# InMemorySaver needs to survive Streamlit re-runs
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session_1"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- TOOL DEFINITIONS ---
@st.cache_resource
def get_tools():
    weather_wrapper = OpenWeatherMapAPIWrapper()
    
    @tool
    def get_weather(location: str) -> str:
        """Fetches the current weather information for a specified location."""
        return weather_wrapper.run(location)

    tavily_tool = TavilySearch(max_results=3, topic="general")
    tavily_tool.description = "Search Tavily for real-time agricultural market prices and trends."

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    wiki_tool.description = "Search Wikipedia for detailed crop facts and botanical information."

    return [get_weather, tavily_tool, wiki_tool]

tools = get_tools()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌾 Krishi Setu Settings")
    st.markdown("---")
    
    model_choice = st.selectbox(
        "Choose AI Brain:",
        ["Qwen (Deep Reasoning)", "Llama (Fast Action)"],
        index=0
    )
    
    active_model_id = "qwen/qwen3-32b" if "Qwen" in model_choice else "llama-3.1-8b-instant"
    
    st.info(f"Using: {active_model_id}")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit_session_{os.urandom(4).hex()}"
        st.rerun()

    st.markdown("---")
    st.markdown("""
        **How to use:**
        - Ask about weather in your region.
        - Check current crop prices.
        - Get organic farming advice.
    """)

# --- AGENT SETUP ---
@st.cache_resource
def get_agent(model_id):
    llm = init_chat_model(model_id, model_provider="groq")
    
    system_prompt = """You are Krishi Setu, an expert agricultural advisor.

### INTELLIGENT ROUTING:
1. If you know the answer from internal knowledge and it's general, answer directly.
2. If the user asks for current weather, use the 'get_weather' tool.
3. If they ask for live prices, current events, or real-time data, use 'tavily_search'.
4. If they need deep historical or botanical facts, use 'wikipedia'.

### OUTPUT FORMATTING:
After receiving data from any tool, process it and provide your final answer formatted strictly using the **GROW model**:
- **Goal**: The farmer's objective.
- **Reality**: Established facts from tools (weather, prices, etc.).
- **Obstacles**: Potential risks identified.
- **Way Forward**: Clear, actionable advice.

Be concise and professional."""

    return create_agent(
        model=llm,
        tools=tools,
        checkpointer=st.session_state.checkpointer,
        system_prompt=system_prompt
    )

agent = get_agent(active_model_id)

# --- CHAT UI ---
st.title("🚜 Krishi Setu - Smart Farm Advisor")
st.subheader("GROW Model")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What would you like to ask Krishi Setu?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Initial status
        status_text = st.status("Thinking...", expanded=False)
        
        try:
            # We use stream to show the flow
            for step in agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                stream_mode="values",
                config=config
            ):
                msg = step["messages"][-1]
                
                if isinstance(msg, AIMessage):
                    # Check for reasoning (Qwen)
                    if hasattr(msg, 'additional_kwargs') and 'reasoning_content' in msg.additional_kwargs:
                        reasoning = msg.additional_kwargs['reasoning_content']
                        with st.expander("Thought Process (Qwen Reasoning)"):
                            st.markdown(reasoning)
                    elif "<think>" in msg.content:
                        # Sometimes reasoning is in content for some providers
                        import re
                        think_match = re.search(r'<think>(.*?)</think>', msg.content, re.DOTALL)
                        if think_match:
                            with st.expander("Thought Process"):
                                st.markdown(think_match.group(1))
                            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

                    # Handle tool calls display in status
                    if msg.tool_calls:
                        tool_names = [tc['name'] for tc in msg.tool_calls]
                        status_text.update(label=f"Using tools: {', '.join(tool_names)}", state="running")
                    
                    if not msg.tool_calls and msg.content:
                        full_response = msg.content
                        status_text.update(label="Complete!", state="complete", expanded=False)
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            status_text.update(label="Failed!", state="error")

st.markdown("---")

