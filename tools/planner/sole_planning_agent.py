import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from project_agents.prompts import planner_agent_prompt # direct strategy prompt only
# from utils.func import get_valid_name_city,extract_before_parenthesis, extract_numbers_from_filenames
import json
import time
import datetime # Import datetime module
# from langchain.callbacks import get_openai_callback # Remove langchain callback if not used with smolagents

from tqdm import tqdm
# from tools.planner.apis import Planner, ReactPlanner, ReactReflectPlanner # Removed old planner classes
# import openai
import argparse
from datasets import load_dataset
from typing import Any, List, Dict

# Add smolagents imports
try:
    from smolagents import CodeAgent, LiteLLMModel
except ImportError:
    print("smolagents not installed. Please run `uv add smolagents litellm`")
    CodeAgent = None
    LiteLLMModel = None

# Add openai-agents imports
try:
    from agents import Agent as OpenAIAgent, Runner, Usage as OpenAIUsage, ModelSettings as OpenAIModelSettings
except ImportError:
    print("openai-agents not installed. Please run `uv add openai-agents`")
    OpenAIAgent = None
    Runner = None
    OpenAIUsage = None
    OpenAIModelSettings = None

# Add langgraph imports
try:
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
except ImportError:
    print("langgraph or langchain-openai not installed. Please run `uv add langgraph langchain-openai`")
    create_react_agent = None
    ChatOpenAI = None

# Add pydantic-ai imports
try:
    from pydantic_ai import Agent as PydanticAgent
except ImportError:
    print("pydantic-ai not installed. Please run `uv add pydantic-ai`")
    PydanticAgent = None

# Add agno imports
try:
    from agno.agent import Agent as AgnoAgent, RunResponse as AgnoRunResponse
    from agno.models.openai import OpenAIChat as AgnoOpenAIChat
    from agno.tools.reasoning import ReasoningTools
except ImportError:
    print("agno not installed. Please run `uv add agno`")
    AgnoAgent = None
    AgnoRunResponse = None
    AgnoOpenAIChat = None
    ReasoningTools = None

# Add autogen imports
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError:
    print("autogen not installed. Please run `uv add autogen-agentchat autogen-ext[openai]`")
    AssistantAgent = None
    TextMessage = None
    OpenAIChatCompletionClient = None

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers

def smolagents_output_to_string(output: Any) -> str:
    """
    Convert smolagents `agent.run` output (AgentText, AgentImage, AgentAudio, etc.)
    into a single unified string representation.
    """
    # If the object provides a to_string() method, call it
    to_string_method = getattr(output, "to_string", None)
    if callable(to_string_method):
        return to_string_method()

    # If the output is a sequence, convert each element recursively and concatenate
    if isinstance(output, (list, tuple)):
        return "".join(smolagents_output_to_string(item) for item in output)

    # Otherwise, fall back to the built-in str()
    return str(output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="openai/gpt-4.1-2025-04-14", help="Model name for the LLM (e.g., 'openai/gpt-4o-mini', 'openai/gpt-4.1-2025-04-14').")
    parser.add_argument("--output_dir", type=str, default="./")
    # parser.add_argument("--strategy", type=str, default="direct") # Strategy fixed to direct
    parser.add_argument("--agent_framework", type=str, default="smolagents", choices=["smolagents", "openai_agents", "langgraph", "pydanticai", "agno", "agno_reasoning", "autogen"], help="Agent framework to use.") # Add agent_framework argument
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for the LLM.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for the LLM.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens for the LLM.")
    args = parser.parse_args()
    directory = f'{args.output_dir}/{args.set_type}'
    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
    numbers = [i for i in range(1,len(query_data_list)+1)]

    # Strategy fixed to direct
    strategy = "direct"

    # Initialize token counters
    total_input_tokens = 0
    total_output_tokens = 0

    for number in tqdm(numbers[:]):

        # Initialize agent based on framework
        agent = None
        if args.agent_framework == "smolagents":
            if CodeAgent is None or LiteLLMModel is None:
                 print("Error: smolagents package not found or classes missing.")
                 sys.exit(1)
            llm_model = LiteLLMModel(
                model_id=args.model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            # Use the direct strategy prompt (planner_agent_prompt) for system prompt/instructions
            agent = CodeAgent(
                tools=[], # No tools defined for sole planning
                model=llm_model,
            )
            print(f"Initialized smolagents CodeAgent with model: {args.model_name}")
        elif args.agent_framework == "openai_agents":
            if OpenAIAgent is None or Runner is None or OpenAIModelSettings is None:
                print("Error: openai-agents package not found or classes missing.")
                sys.exit(1)
            # Instructions are derived from the prompt format later
            agent = OpenAIAgent( # Use the aliased name
                name="TravelPlannerAgent",
                model=args.model_name,
                model_settings=OpenAIModelSettings(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
            )
            print(f"Initialized OpenAI Agent with model: {args.model_name}")
        elif args.agent_framework == "langgraph":
            if ChatOpenAI is None or create_react_agent is None:
                print("Error: langgraph/langchain-openai package not found or classes missing.")
                sys.exit(1)

            # Process model name for langgraph with ChatOpenAI
            processed_model_name = args.model_name
            if "/" in args.model_name:
                prefix, suffix = args.model_name.split("/", 1)
                if prefix == "openai":
                    processed_model_name = suffix
                else:
                    # Potentially handle other prefixes or raise error if unsupported
                    print(f"Warning: Using langgraph with non-openai prefix model: {args.model_name}")
                    processed_model_name = args.model_name # Use original name if not openai

            if prefix == "openai":
                llm = ChatOpenAI(
                    model=processed_model_name,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
            else:
                # Handle initialization for other LLM providers if needed
                print(f"Error: Unsupported model provider for langgraph in this script: {prefix}")
                sys.exit(1)

            # LangGraph's ReAct agent doesn't explicitly take system instructions in the same way.
            # Tools are empty for this task.
            agent = create_react_agent(llm, tools=[])
            print(f"Initialized LangGraph ReAct Agent with model: {processed_model_name}")
        elif args.agent_framework == "pydanticai":
            if PydanticAgent is None:
                print("Error: pydantic-ai package not found.")
                sys.exit(1)

            # Process model name for pydantic-ai (e.g., openai/gpt-4o-mini -> openai:gpt-4o-mini)
            processed_model_id = args.model_name
            if "/" in args.model_name:
                prefix, suffix = args.model_name.split("/", 1)
                if prefix == "openai":
                    processed_model_id = f"{prefix}:{suffix}"
                else:
                    print(f"Warning: Using pydantic-ai with non-openai prefix model: {args.model_name}")

            agent = PydanticAgent(
                model=processed_model_id,
                model_settings={
                    'temperature': args.temperature,
                    'max_tokens': args.max_tokens,
                    'top_p': args.top_p,
                },
            )
            print(f"Initialized PydanticAI Agent with model: {processed_model_id}")
        elif args.agent_framework == "agno" or args.agent_framework == "agno_reasoning":
            if AgnoAgent is None or AgnoOpenAIChat is None:
                print("Error: agno package not found or classes missing.")
                sys.exit(1)

            # Process model name for agno
            processed_model_name = args.model_name
            if "/" in args.model_name:
                prefix, suffix = args.model_name.split("/", 1)
                if prefix == "openai":
                    processed_model_name = suffix
                else:
                    # Assuming non-openai prefixes might be handled directly or raise error later
                    print(f"Warning: Using agno with non-openai prefix model: {args.model_name}")

            # Currently only supporting OpenAI models with agno in this script
            if prefix == "openai":
                llm = AgnoOpenAIChat(
                    id=processed_model_name,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                )
            else:
                print(f"Error: Unsupported model provider for agno in this script: {prefix}")
                sys.exit(1)

            agno_tools = []
            use_reasoning = args.agent_framework == "agno_reasoning"
            if use_reasoning:
                if ReasoningTools is not None:
                    agno_tools.append(ReasoningTools(add_instructions=True))
                    print("Using Agno Agent with ReasoningTools")
                else:
                    print("Error: ReasoningTools selected but not available. Please install `agno[reasoning]`.")
                    sys.exit(1)

            agent = AgnoAgent(
                model=llm,
                tools=agno_tools, # Use the dynamically created list of tools
            )
            print(f"Initialized Agno Agent with model: {processed_model_name}{' and ReasoningTools' if use_reasoning else ''}")
        elif args.agent_framework == "autogen":
            if AssistantAgent is None or TextMessage is None or OpenAIChatCompletionClient is None:
                print("Error: autogen packages not found or classes missing.")
                sys.exit(1)

            # Process model name for AutoGen (expects 'openai/gpt-...' or similar)
            processed_model_name_for_client = args.model_name
            if "/" in args.model_name:
                prefix, suffix = args.model_name.split("/", 1)
                if prefix != "openai": # Currently only support openai prefix
                    print(f"Warning: Unsupported model prefix for autogen: {prefix}. Using original.")
                else:
                    # Use the suffix for OpenAIChatCompletionClient
                    processed_model_name_for_client = suffix

            if prefix == "openai":
                model_client = OpenAIChatCompletionClient(
                    model=processed_model_name_for_client,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                )
            else:
                print(f"Error: Unsupported model provider for autogen in this script: {prefix}")
                sys.exit(1)

            # AutoGen doesn't have a direct concept of 'system prompt' in the same way as some other frameworks for simple AssistantAgent.
            # We'll pass the formatted prompt as the initial message.
            agent = AssistantAgent(
                name="TravelPlannerAgent",
                model_client=model_client,
            )
            print(f"Initialized AutoGen AssistantAgent with model: {args.model_name}")

        else:
            print(f"Error: Unsupported agent_framework '{args.agent_framework}'")
            sys.exit(1)

        query_data = query_data_list[number-1]
        reference_information = query_data['reference_information']
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        while True:
            # Format the prompt string - used differently depending on the framework
            prompt_text = planner_agent_prompt.format(text=reference_information, query=query_data['query'])

            # Call the agent
            if args.agent_framework == "smolagents":
                # Run smolagents agent
                agent_response = agent.run(prompt_text, reset=True)
                planner_results = smolagents_output_to_string(agent_response)

                # Accumulate token usage
                token_counts = agent.monitor.get_total_token_counts()
                token_usage["input_tokens"] = token_counts.get("input")
                token_usage["output_tokens"] = token_counts.get("output")
            elif args.agent_framework == "openai_agents":
                # Run openai_agents agent
                agent_response = Runner.run_sync(agent, prompt_text)
                planner_results = agent_response.final_output

                # Accumulate token usage
                for resp in agent_response.raw_responses:
                    token_usage["input_tokens"] = resp.usage.input_tokens
                    token_usage["output_tokens"] = resp.usage.output_tokens
            elif args.agent_framework == "langgraph":
                # Run langgraph agent
                agent_response = agent.invoke({"messages": [{"role": "user", "content": prompt_text}]})
                agent_message_dict = agent_response["messages"][-1]
                planner_results = agent_message_dict.content

                # Accumulate token usage
                usage = agent_message_dict.response_metadata["token_usage"]
                token_usage["input_tokens"] = usage.get('prompt_tokens')
                token_usage["output_tokens"] = usage.get('completion_tokens')
            elif args.agent_framework == "pydanticai":
                # Run pydantic-ai agent
                # PydanticAI agent.run() expects user_prompt
                agent_response = agent.run_sync(user_prompt=prompt_text)
                planner_results = agent_response.output

                # Accumulate token usage
                usage = agent_response.usage()
                token_usage["input_tokens"] = usage.request_tokens
                token_usage["output_tokens"] = usage.response_tokens
            elif args.agent_framework == "agno" or args.agent_framework == "agno_reasoning":
                # Run agno agent
                agent_response: AgnoRunResponse = agent.run(prompt_text)
                planner_results = agent_response.content

                # Accumulate token usage
                usage = agent_response.metrics
                token_usage["input_tokens"] = usage.get('prompt_tokens')
                token_usage["output_tokens"] = usage.get('completion_tokens')
            elif args.agent_framework == "autogen":
                # Run autogen agent
                # AutoGen AssistantAgent's run expects a single string task
                # It doesn't maintain history internally in the simple run case.
                # We send the combined prompt each time.
                agent_response = agent.run(task=prompt_text)
                planner_results = agent_response.messages[-1].content # Assuming last message is the response

                # Accumulate token usage
                usage = agent_response.messages[-1].models_usage # Assuming usage is in the last message
                token_usage["input_tokens"] = usage.prompt_tokens
                token_usage["output_tokens"] = usage.completion_tokens
            else:
                print(f"Error: Unsupported agent_framework '{args.agent_framework}'")
                sys.exit(1)

            if planner_results != None:
                break

        # Get token counts
        total_input_tokens += token_usage["input_tokens"]
        total_output_tokens += token_usage["output_tokens"]

        print(planner_results)
        # check if the directory exists
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
            os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
            result =  [{}]
        else:
            result = json.load(open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')))
        # if args.strategy in ['react','reflexion']:
        #     result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results_logs'] = scratchpad 
        # Replace '/' with '-' in model name for the result key
        model_name_for_key = args.model_name.replace("/", "-")
        result[-1][f'{args.agent_framework}_{model_name_for_key}_{strategy}_sole-planning_results'] = planner_results
        # write to json file
        with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    # ----- Token Usage Aggregation and Saving ------
    # Prepare data for JSON
    model_name_for_key = args.model_name.replace("/", "-")
    usage_key = f'{args.agent_framework}_{model_name_for_key}_{strategy}_sole-planning_results'
    timestamp = datetime.datetime.now().isoformat() # Get current timestamp
    usage_data = {
        usage_key: {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "data_count": len(numbers), # Number of queries processed
            "timestamp": timestamp # Add timestamp
        }
    }

    # Define JSON file path
    token_usage_file_path = os.path.join(f'{args.output_dir}/{args.set_type}', 'token_usages.json')

    # Read existing data or initialize
    all_usages = []
    if os.path.exists(token_usage_file_path):
        try:
            with open(token_usage_file_path, 'r') as f:
                all_usages = json.load(f)
            if not isinstance(all_usages, list):
                print(f"Warning: Content in {token_usage_file_path} is not a list. Re-initializing.")
                all_usages = []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {token_usage_file_path}. Re-initializing.")
            all_usages = []
        except FileNotFoundError:
            pass # File doesn't exist yet, will be created

    # Append new usage data
    all_usages.append(usage_data)

    # Write updated data back to JSON
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(token_usage_file_path), exist_ok=True)
        with open(token_usage_file_path, 'w') as f:
            json.dump(all_usages, f, indent=4, ensure_ascii=False)
        print(f"Token usage saved to {token_usage_file_path}")
    except Exception as e:
        print(f"Error saving token usage to {token_usage_file_path}: {e}")

