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
    parser.add_argument("--agent_framework", type=str, default="smolagents", choices=["smolagents", "openai_agents"], help="Agent framework to use.") # Add agent_framework argument
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
    total_usage_openai = None # Initialize usage counter for openai_agents

    # Initialize total usage counter for openai_agents if needed
    if args.agent_framework == "openai_agents":
        if OpenAIUsage is None:
            print("Error: openai-agents package not found or OpenAIUsage class missing.")
            sys.exit(1)
        total_usage_openai = OpenAIUsage()

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
        else:
            print(f"Error: Unsupported agent_framework '{args.agent_framework}'")
            sys.exit(1)

        query_data = query_data_list[number-1]
        reference_information = query_data['reference_information']
        agent_response = None
        while True:
            # Format the prompt string - used differently depending on the framework
            prompt_text = planner_agent_prompt.format(text=reference_information, query=query_data['query'])

            # Call the agent
            if args.agent_framework == "smolagents":
                agent_response = agent.run(prompt_text, reset=True)
                planner_results = smolagents_output_to_string(agent_response)
            elif args.agent_framework == "openai_agents":
                agent_response = Runner.run_sync(agent, prompt_text)
                planner_results = agent_response.final_output

                # Accumulate token usage for this run
                if total_usage_openai is not None:
                    for resp in agent_response.raw_responses:
                            if hasattr(resp, 'usage') and resp.usage is not None:
                                total_usage_openai.add(resp.usage)
            else:
                print(f"Error: Unsupported agent_framework '{args.agent_framework}'")
                sys.exit(1)

            if planner_results != None:
                break

        # Get token counts
        if args.agent_framework == "smolagents":
            token_counts = agent.monitor.get_total_token_counts()
            total_input_tokens += token_counts.get("input")
            total_output_tokens += token_counts.get("output")
        elif args.agent_framework == "openai_agents" and total_usage_openai is not None:
            for resp in agent_response.raw_responses:
                if hasattr(resp, 'usage') and resp.usage is not None:
                    total_input_tokens += resp.usage.input_tokens
                    total_output_tokens += resp.usage.output_tokens

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

