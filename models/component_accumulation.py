import os
import json
import argparse
from tqdm import tqdm
from utils import OpenAIModel


class ComponentAccumulator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.exp_description = args.exp_description
        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}/stack.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()

    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.exp_description}_reordered.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def prompt_creator(self, sentence_structure_pairs):
        full_prompt = self.prompt_template.replace('[[SENTENCE_STRUCTURE_PAIRS]]', sentence_structure_pairs)
        return full_prompt
    
    def batch_component_accumulation(self, batch_size = 10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset()
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = []
            full_info = []
            for example in chunk:
                sentence_structure_pairs_list = example[f"{self.exp_description}_structure"]
                if sentence_structure_pairs_list is None:
                    sentence_structure_pairs_list = ["None"]
                for sentence_structure_pairs in sentence_structure_pairs_list:
                    full_prompts.append(self.prompt_creator(sentence_structure_pairs))
                    full_info.append(example["id"])
            batch_outputs = self.openai_api.batch_generate(full_prompts)

            id_tmp = 'none'
            output_tmp = []
            code_list = []
            for info, output in zip(full_info, batch_outputs):
                if info != id_tmp:
                    if len(output_tmp) != 0:
                        code_list.append(output_tmp)
                        output_tmp = []
                    id_tmp = info
                output_tmp.append(output)
            code_list.append(output_tmp)

            assert len(chunk) == len(code_list)
            for sample, code in zip(chunk, code_list):
                output = {
                    'id': sample['id'],
                    'context': sample['context'],
                    'question': sample['question'],
                    'options': sample['options'],
                    'answer': sample['answer'],
                    'declarations': sample['declarations'],
                    'constraints': sample['constraints'],
                    'queries': sample['queries'],
                    'additional_constraints': sample['additional_constraints'],
                    'queries_function': sample['queries_function'],
                    f'{self.exp_description}_structure': sample[f'{self.exp_description}_structure'],
                    f'{self.exp_description}_accumulation': code,
                }
                outputs.append(output)

        # remove examples with duplicate ids from the result
        outputs = list({output['id']: output for output in outputs}.values())
        print(f"Generated {len(outputs)} examples.")

        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.exp_description}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./outputs/logical_dependency_structure')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/component_accumulation')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--exp_description', type=str, default='constraints') # constraints, queries
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    component_accumulator = ComponentAccumulator(args)
    component_accumulator.batch_component_accumulation()