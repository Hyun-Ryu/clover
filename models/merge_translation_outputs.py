import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./outputs/first_order_logic_translation')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--model_name', type=str, default='gpt-4o')
args = parser.parse_args()

dir_ = args.data_path

# translation output files
constraints_ = f'{args.dataset_name}_{args.split}_{args.model_name}_constraints_reordered.json'
queries_ = f'{args.dataset_name}_{args.split}_{args.model_name}_queries_reordered.json'
additional_constraints_ = f'{args.dataset_name}_{args.split}_{args.model_name}_additional_constraints_reordered.json'

# ensemble file
outputs = []
ensemble = f'{args.dataset_name}_{args.split}_{args.model_name}_constraints_queries_additional_constraints_reordered.json'

# example ids
with open(os.path.join(dir_, constraints_)) as f:
    raw_dataset = json.load(f)
example_ids = []
for data in raw_dataset:
    example_ids.append(data['id'])

# main
for id in example_ids:
    # constraints
    with open(os.path.join(dir_, constraints_)) as f:
        raw_dataset = json.load(f)
    for data in raw_dataset:
        if data["id"] == id:
            output_dict = {
                "id": data["id"],
                "context": data["context"],
                "question": data["question"],
                "options": data["options"],
                "answer": data["answer"],
                "declarations": data["declarations"],
                "constraints": data["constraints"],
                "queries": data["queries"],
                "additional_constraints": data["additional_constraints"],
                "queries_function": data["queries_function"],
                "constraints_translation": data["constraints_translation"],
                "queries_translation": None,
                "additional_constraints_translation": None,
            }
    
    # queries
    with open(os.path.join(dir_, queries_)) as f:
        raw_dataset = json.load(f)
    for data in raw_dataset:
        if data["id"] == id:
            output_dict["queries_translation"] = data["queries_translation"]

    # additional constraints
    with open(os.path.join(dir_, additional_constraints_)) as f:
        raw_dataset = json.load(f)
    for data in raw_dataset:
        if data["id"] == id:
            if data["additional_constraints"] is not None:
                output_dict["additional_constraints_translation"] = data["additional_constraints_translation"]
    
    outputs.append(output_dict)

# save the ensemble json file
with open(os.path.join(dir_, ensemble), 'w') as f:
    json.dump(outputs, f, indent=2, ensure_ascii=False)