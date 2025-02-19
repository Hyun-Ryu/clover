#!/bin/bash
### FOL Translation ###
python models/logical_dependency_structure.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description constraints
python models/logical_dependency_structure.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description queries
python models/logical_dependency_structure.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description additional_constraints

python models/component_accumulation.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description constraints
python models/component_accumulation.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description queries
python models/component_accumulation.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description additional_constraints

python models/first_order_logic_translation.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description constraints
python models/first_order_logic_translation.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description queries
python models/first_order_logic_translation.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description additional_constraints

### Merge constraints, queries, and additional constraints ###
python merge_translation_outputs.py --dataset_name AR-LSAT --split dev --model_name ${MODEL_NAME}

### FOL Verification ###
python models/first_order_logic_verification.py --dataset_name AR-LSAT --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --verification_method ${VERIFICATION} --exp_descriptions constraints queries additional_constraints

### Execute logic programs ###
python models/logic_inference.py --dataset_name AR-LSAT --split dev --model_name ${MODEL_NAME} --verification_method ${VERIFICATION} --backup_strategy random