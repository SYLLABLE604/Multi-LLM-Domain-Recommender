import json
import logging
import os
import queue
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.instruction_generate import demon_prompt_generate, task_instruction_generate

import argparse
from src.naive_model_thread import NaiveModelThread
from src.model_load import load_model
from src.assist_model_thread import AssistModelThread


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some files.')

    parser.add_argument('--config', help='the name of the file to process')
    parser.add_argument('--result_save_dir', '-rsd', default="./", type=str, required=False, help='result_save_dir')
    parser.add_argument('--run_mode', '-rm', default="dev", type=str, required=False, help='result_save_dir')
    parser.add_argument('--logits_processor_mode', '-lpm', default="based_on_probility_transfer_logits_local_processor",
                        type=str,
                        required=False,
                        help='logits_processor_mode')
    parser.add_argument('--device_compute', '-dp', default="cuda:0", type=str, required=False,
                        help='device_compute')
    parser.add_argument('--device0', '-d0', default="cuda:0", type=str, required=False,
                        help='device0')
    parser.add_argument('--device1', '-d1', default="cuda:1", type=str, required=False,
                        help='device1')
    parser.add_argument('--device2', '-d2', default="cuda:2", type=str, required=False,
                        help='device2')
    parser.add_argument('--device3', '-d3', default="cuda:3", type=str, required=False,
                        help='device3')
    parser.add_argument('--device4', '-d4', default="cuda:4", type=str, required=False,
                        help='device4')
    parser.add_argument('--device5', '-d5', default="cuda:5", type=str, required=False,
                        help='device5')
    parser.add_argument('--device6', '-d6', default="cuda:6", type=str, required=False,
                        help='device6')
    parser.add_argument('--device7', '-d7', default="cuda:7", type=str, required=False,
                        help='device7')
    parser.add_argument('--device8', '-d8', default="cuda:0", type=str, required=False,
                        help='device8')

    parser.add_argument('--ensemble_weight', '-ew',
                        nargs='+',
                        type=float,
                        default=[1.0], help='ensemble_weight', required=False
                        )

    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config_json = json.load(f)

    model_paths = config_json["model_path"]
    assist_model_count = len(model_paths) - 1
    
    naive_model_path = config_json["model_path"]["naive_model_path"]
    naive_model_probability_transfer_matrix_path = config_json["probability_transfer_matrix_path"]["naive_model_path"]
    model_system_template = config_json["prompt_template"]["model_system_template"]


    #输入文件，修改为自己的内容
    naive_file_path = config_json["file_path"]["naive_file_path"] #naive模型，没有信息

    instruction = config_json["prompt_template"]["instruction"]
    instruction_parameter = config_json["prompt_template"]["instruction_parameter"]
    max_new_tokens = config_json["run_parameter"]["max_new_tokens"]

    result_process_parameter = config_json["result_process_parameter"]

    try:
        early_stop_string_list = result_process_parameter["early_stop_string_list"]
    except:
        early_stop_string_list = None

    result_save_dir = args.result_save_dir
    logits_processor_mode = args.logits_processor_mode
    if os.path.isdir(result_save_dir):
        pass
    else:
        os.makedirs(result_save_dir)

    device_compute = args.device_compute

    device0 = args.device0
    device1 = args.device1
    device2 = args.device2
    device3 = args.device3
    device_list = [device0, device1, device2, device3]


    logging.basicConfig(filename=os.path.join(result_save_dir,
                                              f'{start_time}.process.log'),
                        level=logging.DEBUG)
    logging.info(f'\n【config_json:】{config_json}')
    logging.info(f'\n【result_save_dir:】{result_save_dir}')


    naive_model, naive_model_tokenizer, naive_model_streamer = load_model(naive_model_path, "auto")
    assist_model_tokenizer_list = []
    naive_model_probability_transfer_matrix_list = []
    assist_model_probability_transfer_matrix_list = []

    if assist_model_count != 0:
        naive_model_probability_transfer_matrix = torch.load(naive_model_probability_transfer_matrix_path,
                                                            map_location=device_compute)
        naive_model_probability_transfer_matrix_list = [naive_model_probability_transfer_matrix]

        assist_model_list = []
        assist_model_tokenizer_list = []
        assist_model_system_template_list = []
        assist_model_probability_transfer_matrix_list = []
        assist_model_probability_inverse_transfer_matrix_list = []

        for index in range(1, assist_model_count + 1):
            assist_model, assist_model_tokenizer, _ = load_model(
                config_json["model_path"]["model" + str(index) + "_path"], "auto")

            assist_model_list.append(assist_model)
            assist_model_tokenizer_list.append(assist_model_tokenizer)
            assist_model_system_template_list.append(
                config_json["prompt_template"]["model_system_template"])
            assist_model_probability_transfer_matrix_list.append(
                torch.load(config_json["probability_transfer_matrix_path"]["model" + str(index) + "_path"],
                           map_location=device_list[0]))
            assist_model_probability_inverse_transfer_matrix_list.append(
                torch.load(config_json["probability_transfer_inv_matrix_path"]["model" + str(index) + "_path"],
                           map_location=device_list[0]))

    # ================================================================

    start_index = 0
    # result_file_path = os.path.join(result_save_dir,
    #                                 f'ensemble_lr{learning_rate}_learning_epochs_nums{learning_epochs_nums}.jsonl')
    # try:
    #     with open(result_file_path, 'r') as file:
    #         lines = file.readlines()
    #         line_count = len(lines)
    #     start_index = line_count
    # except:
    #     start_index = 0

    assit_input_list = []
    for index in range(1, assist_model_count + 1):
        with open(config_json["file_path"][f"file{index}_path"], 'r', encoding='utf-8') as naive_file:
            assist_contents = naive_file.readlines()
            assit_input_list.append(assist_contents[start_index:])

    with open(naive_file_path, 'r', encoding='utf-8') as naive_file:
        contents = naive_file.readlines()
        
        for text_index, line in enumerate(tqdm(contents[start_index:])):
            line = json.loads(line)
            # print(line)
            # print(instruction_parameter)
            task_instruction = task_instruction_generate(line, instruction_parameter)
            # trunk_num = line['trunk_num']
            naive_input_prompt = instruction + task_instruction
            naive_model_input = model_system_template.format(naive_input_prompt)

            information_dict = {}
            information_dict['main_model_input'] = naive_model_input
            information_dict['demon_count'] = 0
            information_dict['task_instruction'] = task_instruction
            information_dict['max_new_tokens'] = max_new_tokens
            information_dict['result_process_parameter'] = result_process_parameter
            information_dict['logits_processor_mode'] = logits_processor_mode
            # information_dict['trunk_num'] = trunk_num

            ensemble_model_output_ids_queue = queue.Queue()

            assist_model_score_queue_list = []
            assist_model_input_list = []
            assist_model_trunk_list = []

            for assist_index in range(0, assist_model_count):
                assist_model_score_queue_list.append(queue.Queue())
                # print(assit_input_list[assist_index][text_index])
                # print(instruction_parameter)
                text = json.loads(assit_input_list[assist_index][text_index])
                assist_task_instruction = task_instruction_generate(text, instruction_parameter)
                assist_input_prompt = instruction + assist_task_instruction
                assist_model_input_list.append(assist_model_system_template_list[assist_index].format(assist_input_prompt))
                assist_model_trunk_list.append(text['trunk_num'])
            print(information_dict['main_model_input'])
            print(assist_model_input_list)
            print(assist_model_trunk_list)

            
            output = ""
            weight_list = []
            aggrated_distance_to_native_list = []
            answer = ""
            for i in range(max_new_tokens):
                main_model_thread = NaiveModelThread(main_model=naive_model,
                                                main_model_tokenizer=naive_model_tokenizer,
                                                assist_model_tokenizer=assist_model_tokenizer_list,
                                                information_dict=information_dict,
                                                result_save_dir=result_save_dir,
                                                ensemble_model_output_ids_queue=ensemble_model_output_ids_queue,
                                                assist_model_score_queue_list=assist_model_score_queue_list,
                                                main_model_probability_transfer_matrix_list=naive_model_probability_transfer_matrix_list,
                                                assist_model_probability_transfer_matrix_list=assist_model_probability_transfer_matrix_list,
                                                assist_model_probability_inverse_transfer_matrix_list=assist_model_probability_inverse_transfer_matrix_list,
                                                device_compute=device_compute,
                                                device=device0,
                                                early_stop_string_list=early_stop_string_list
                                                )
                main_model_thread.start()
                # print("main model started")

                assist_model_thread_list = []
                for index in range(0, assist_model_count):
                    # print(assist_model_list[index])
                    assist_model_thread = AssistModelThread(model=assist_model_list[index],
                                                            model_tokenizer=assist_model_tokenizer_list[index],
                                                            assist_model_input=assist_model_input_list[index],
                                                            assist_model_score_queue=assist_model_score_queue_list[
                                                                index],
                                                            device=device_list[0],
                                                            result_save_dir=result_save_dir
                                                            )
                    assist_model_thread.start()
                    assist_model_thread_list.append(assist_model_thread)
                # print("assist model started")
                for assist_model_thread in assist_model_thread_list:
                    assist_model_thread.join()
               
                
                
                if max_new_tokens != 1:
                    
                    try:
                        #得到主模型的logits
                        print(f"-------------------------------------")
                        print(f"round {i}")
                        output = ensemble_model_output_ids_queue.get(block=True,
                            timeout=600)
                        # logging.info(
                        #     f'{i}, {output}')
                        print(i, output['next_tokens'])
                    except:
                        print(f"cooporation end")
                        output = information_dict['main_model_input']
                        break

                    temp_tokens = output['next_tokens']
                    weight_list.append(output['weight_list'])
                    aggrated_distance_to_native_list.append(output['aggrated_distance_to_native'])

                    if isinstance(temp_tokens[0], bytes):
                        temp_tokens = temp_tokens.decode("utf-8")
                    if temp_tokens.startswith('▁'):
                        new_token = " " + temp_tokens[1:]
                    else:
                        new_token = temp_tokens

                    if new_token == "</s>":
                        output = information_dict['main_model_input'] + new_token
                        break
                    else:
                        weight_list.append(output['weight_list'])
                        aggrated_distance_to_native_list.append(output['aggrated_distance_to_native'])

                    #更新模型的input_list
                    for index in range(len(assist_model_input_list)):
                        assist_model_input_list[index] += new_token
                        # print(f"assist model {index} round {i+1} input: {assist_model_input_list[index]}")
                    information_dict['main_model_input'] += new_token
                    answer += new_token
                    # print(f"main model round {i+1} input: {information_dict['main_model_input']}")
                main_model_thread.join()
                
            aggrated_distance_sum = sum(aggrated_distance_to_native_list)
            for index in range(0, assist_model_count):
                value = 0
                for weight, aggrated_distance in zip(weight_list, aggrated_distance_to_native_list):
                    # print(weight)
                    # print(aggrated_distance)
                    value += weight[index] * aggrated_distance / aggrated_distance_sum
                logging.info(f"For question {text_index}. Assist model {index}, trunk number {assist_model_trunk_list[index]}, output value: {value}")
            logging.info(f"---------------------------------")
            # logging.info(f"output: {answer}")
                # print(f"For question {text_index}. Assist model {index}, trunk number {assist_model_trunk_list[index]}, output value: {value}")


            # print(f"output: {answer}")
            


    time_elapsed = time.time() - start_time  # 获得时间差
    minutes = int(time_elapsed / 60)
    seconds = int(time_elapsed % 60)
    logging.info(f"\nTime taken: {minutes} min {seconds} sec")
    print('Time taken: {} min {} sec'.format(minutes, seconds))


if __name__ == '__main__':
    main()
