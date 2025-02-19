#!/bin/bash
### FOL Translation ###
python models/logical_dependency_structure.py --dataset_name ZebraLogic --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description constraints
python models/component_accumulation.py --dataset_name ZebraLogic --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description constraints --max_new_tokens 2048
python models/first_order_logic_translation.py --dataset_name ZebraLogic --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --exp_description constraints --max_new_tokens 4096

### FOL Verification ###
python models/first_order_logic_verification.py --dataset_name ZebraLogic --split dev --api_key ${API_KEY} --model_name ${MODEL_NAME} --verification_method ${VERIFICATION} --exp_descriptions constraints

### Execute logic programs ###
python models/logic_inference.py --dataset_name ZebraLogic --split dev --model_name ${MODEL_NAME} --backup_strategy random --verification_method ${VERIFICATION}