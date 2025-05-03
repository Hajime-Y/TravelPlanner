import argparse
from datasets import load_dataset
from tqdm import tqdm
import json


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")
    parser.add_argument("--agent_framework", type=str, default=None, choices=["smolagents", "openai_agents", "langgraph", "pydanticai"], help="Specify the agent framework used for sole-planning agent mode.")

    args = parser.parse_args()

    model_name_for_path = args.model_name.replace("/", "-")
    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        if args.agent_framework:
            suffix = f'_{args.agent_framework}_{args.strategy}'
        else:
            suffix = f'_{args.strategy}'

    results_file_path = f'{args.tmp_dir}/{args.set_type}_{model_name_for_path}{suffix}_{args.mode}.txt'
    results = open(results_file_path,'r').read().strip().split('\n')

    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']

    idx_number_list = [i for i in range(1,len(query_data_list)+1)]
    for idx in tqdm(idx_number_list[:]):
        generated_plan = json.load(open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json'))

        if args.mode == 'sole-planning' and args.agent_framework:
            model_name_for_key = args.model_name.replace("/", "-")
            result_key = f'{args.agent_framework}_{model_name_for_key}_{args.strategy}_sole-planning_results'
            parsed_result_key = f'{args.agent_framework}_{model_name_for_key}_{args.strategy}_sole-planning_parsed_results'
        else:
            result_key = f'{args.model_name}{suffix}_{args.mode}_results'
            parsed_result_key = f'{args.model_name}{suffix}_{args.mode}_parsed_results'

        if generated_plan[-1][result_key] not in ["","Max Token Length Exceeded."] :
            try:
                # result = results[idx-1].split('```json')[1].split('```')[0]
                # no need to split with ```json and ```
                result = results[idx-1]
                if '\t' in result:
                    result = result.split('\t', 1)[1]
            except:
                print(f"{idx}:\n{results[idx-1]}\nThis plan cannot be parsed. The plan has to follow the format ```json [The generated json format plan]```(The common gpt-4-preview-1106 json format). Please modify it manualy when this occurs.")
                break
            try:
                generated_plan[-1][parsed_result_key] =  eval(result)
            except:
                print(f"{idx}:\n{result}\n This is an illegal json format. Please modify it manualy when this occurs.")
                break
        else:
            generated_plan[-1][parsed_result_key] = None

        with open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json','w') as f:
            json.dump(generated_plan,f)