import json
import math
import os
import queue
from collections import Counter

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from transformers import LogitsProcessor


class BasedOnProbabilityTransferLogits_Naive_Processor(LogitsProcessor):
    def __init__(self,
                 ensemble_model_output_ids_queue, assist_model_score_queue_list,
                 main_model_probability_transfer_matrix_list,
                 assist_model_probability_transfer_matrix_list, assist_model_probability_inverse_transfer_matrix_list, 
                 result_save_dir, main_model_tokenizer,
                 assist_model_tokenizer, device, device_compute, early_stop_string_list=None):
        self.assist_model_score_queue_list = assist_model_score_queue_list
        self.ensemble_model_output_ids_queue = ensemble_model_output_ids_queue
        self.main_model_probability_transfer_matrix_list = main_model_probability_transfer_matrix_list
        self.assist_model_probability_transfer_matrix_list = assist_model_probability_transfer_matrix_list
        self.assist_model_probability_inverse_transfer_matrix_list = assist_model_probability_inverse_transfer_matrix_list
        self.result_save_dir = result_save_dir
        self.main_model_tokenizer = main_model_tokenizer
        self.assist_model_tokenizer_list = assist_model_tokenizer
        self.device = device
        self.device_compute = device_compute
        self.early_stop_string_list = early_stop_string_list

        self.target_assist_model = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ensemble_process_file_path = os.path.join(self.result_save_dir,
                                                  f'ensemble_lr_anchor_point_count_all_learning_epochs_nums_5.log')
        main_model_only_flag = False
        json_object = {}

        assist_model_generate_ids_logits_list = []
        for index, queue_instance in enumerate(self.assist_model_score_queue_list):
            try:
                value = queue_instance.get(block=True, timeout=20)
                assist_model_generate_ids_logits_list.append(value)

            except queue.Empty:
                print(f"aux model{index}【not received】\n")
                assist_model_generate_ids_logits_list.append(None)
                main_model_only_flag = True

        # print(f"len of assist output: {len(assist_model_generate_ids_logits_list)}")
        if len(assist_model_generate_ids_logits_list) == 0:
            main_model_only_flag = True
        if torch.argmax(scores).item() == self.main_model_tokenizer.eos_token_id:
            main_model_only_flag = True

        if self.early_stop_string_list is not None:
            for early_stop_string in self.early_stop_string_list:
                early_stop_token = self.main_model_tokenizer(early_stop_string, return_tensors="pt",
                                                             add_special_tokens=False).input_ids.tolist()[0][1:]
                last_token_count = len(early_stop_token)

                last_token_ids = input_ids.tolist()[0][-last_token_count:]
                if last_token_ids == early_stop_token:
                    scores[:, self.main_model_tokenizer.eos_token_id] = float('inf')
                    main_model_only_flag = True
        
        #和子模型进行交互的逻辑
        if not main_model_only_flag:

            main_model_generate_ids_logits = Variable(scores, requires_grad=True).to(torch.float32).to(
                self.main_model_probability_transfer_matrix_list[0].device)

            with torch.no_grad():
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits,
                                                                      dim=-1)
                # main_model_generate_ids_probs = main_model_generate_ids_logits
                #原始输出采样
                main_model_generate_ids_probs_values, main_model_generate_ids_probs_indices = torch.topk(
                    main_model_generate_ids_probs, k=10)
                json_object[f'origin_naive_top_tokens'] = self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_probs_indices.tolist()[0])
                #转移到公共空间
                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[
                                                                        0]).to(self.device_compute)

                main_model_relative_values, main_model_relative_indices = torch.topk(
                    main_model_relative_representation_probs, k=10)
                json_object[f'main_rel_values'] = main_model_relative_values.tolist()[0]
                json_object[f'main_rel_indices'] = main_model_relative_indices.tolist()[0]
                # print(f"main model token: {json_object[f'main_rel_indices']}")
                model_relative_representation_probs_list = []
                #寻找子模型输出
                for index, (assist_model_generate_ids_logits, assist_model_probability_transfer_matrix) in enumerate(
                        zip(assist_model_generate_ids_logits_list,
                            self.assist_model_probability_transfer_matrix_list)):
                    assist_model_generate_ids_probs = nn.functional.softmax(
                        assist_model_generate_ids_logits.to(torch.float32),
                        dim=-1).to(assist_model_probability_transfer_matrix.device)
                    # assist_model_generate_ids_probs = assist_model_generate_ids_logits.to(assist_model_probability_transfer_matrix.device)
                    #子模型原始输出
                    values, indices = torch.topk(assist_model_generate_ids_probs, k=10)
                    json_object[f'origin_aux_{index}_top_tokens'] = self.assist_model_tokenizer_list[
                        index].convert_ids_to_tokens(indices.tolist()[0])
                    #子模型输出映射到公共空间
                    # print(f"shape of output {index} is: {assist_model_generate_ids_probs.shape}")
                    # print(f"shape of transfer matrix {index} is: {assist_model_probability_transfer_matrix.shape}")
                    assist_model_relative_representation_probs = torch.mm(assist_model_generate_ids_probs,
                                                                          assist_model_probability_transfer_matrix).to(
                        self.device_compute)

                    assist_model_relative_values, assist_model_relative_indices = torch.topk(
                        assist_model_relative_representation_probs, k=10)
                    json_object[f'aux_rel_values_{index}'] = assist_model_relative_values.tolist()[0]
                    json_object[f'aux_rel_indices_{index}'] = assist_model_relative_indices.tolist()[0]
                    # print(f"assist model {index} token: {json_object[f'aux_rel_indices_{index}']}")

                    model_relative_representation_probs_list.append(assist_model_relative_representation_probs)
            
            #进行加权，此处修改为计算每个模型输出与naive的距离并加权聚合，
            #计算每个模型输出的价值，
            #选取与聚合后距离最近的模型作为解码模型，输出解码采样后的结果到文件中，
            #主模型也要在test.py文件中使用loop来进行更新prompt并且读取新prompt运行，删除本文件后续的代码

            average_probs = torch.zeros_like(main_model_relative_representation_probs)

            log_main_probs = torch.log(main_model_relative_representation_probs)
            criterion = nn.KLDivLoss()

            value_json_object = {}
            relative_distance_list = []
            aggrated_distance_list = []
            value_list = []
            relative_probs_distance_sum = 0
            aggrated_assist_probs_distance_sum = 0

            weight = []
            #计算聚合向量，log后的向量和没有log的向量有什么区别
            for probs in model_relative_representation_probs_list:
                #计算naive和每个模型输出的loss的距离
                loss = criterion(log_main_probs, probs).item()
                relative_distance_list.append(loss)
                relative_probs_distance_sum += loss
            for distance, probs in zip(relative_distance_list,model_relative_representation_probs_list):
                weight.append(distance / relative_probs_distance_sum)
                average_probs += probs * distance / relative_probs_distance_sum

            #计算token价值
            for probs in model_relative_representation_probs_list:
                aggrated_distance = criterion(average_probs, probs).item()
                aggrated_distance_list.append(aggrated_distance)
                aggrated_assist_probs_distance_sum += aggrated_distance

            aggrated_distance_to_native = criterion(log_main_probs, average_probs).item()

            # for sum_distance, distance in zip(aggrated_distance_list, relative_distance_list):
            #     value = (distance / relative_probs_distance_sum) * (aggrated_distance_to_native / aggrated_assist_probs_distance_sum)
            #     value_list.append(value)
            #聚合后采样,
            average_relative_probs_values, average_relative_probs_indices = torch.topk(
                average_probs, k=10)

            json_object[f'average_rel_probs_values'] = average_relative_probs_values.tolist()[0]
            json_object[f'average_rel_probs_indices'] = average_relative_probs_indices.tolist()[0]
            # json_object[f'relative_information_values'] = value_list
            json_object[f'weight'] = weight

            # print(f"weight {json_object[f'weight']}")
            #向量的反向映射
            target_model_index = aggrated_distance_list.index(min(aggrated_distance_list))
            target_assist_tokenizer = self.assist_model_tokenizer_list[target_model_index]
            target_inv_transfer_matrix = self.assist_model_probability_inverse_transfer_matrix_list[target_model_index]
            #计算逆矩阵或者伪逆矩阵
            try:
                average_real_probs = torch.linalg.solve(target_inv_transfer_matrix, average_probs.T).T.to(self.device_compute)
            except:
                y_pseudo_inv = target_inv_transfer_matrix  # 形状 (3x2)
                average_real_probs = average_probs @ y_pseudo_inv.to(self.device_compute)

            average_real_probs_values, average_real_probs_indices = torch.topk(
                average_real_probs, k=10)
            
            json_object[f'average_tokens'] = target_assist_tokenizer.convert_ids_to_tokens(
                    average_real_probs_indices.tolist()[0])
            
            json_object[f'average_real_probs_values_final'] = average_real_probs_values.tolist()[0]
            json_object[f'average_real_probs_indices_final'] = average_real_probs_indices.tolist()[0]
            
            # target_model_index = average_real_probs_values.tolist()[0].index(max(average_real_probs_values.tolist()[0]))
            # next_tokens = target_assist_tokenizer.convert_ids_to_tokens(target_model_index)
            next_tokens = json_object[f'average_tokens'][0]

            #将最终输出放回主模型序列，需要通过子模型解码后放回主模型序列，直接返回next_token
            # print(f"next_token: {next_tokens}")
            output_json = {}
            output_json['next_tokens'] = next_tokens
            output_json['weight_list'] = weight
            output_json['aggrated_distance_to_native'] = aggrated_distance_to_native

            self.ensemble_model_output_ids_queue.put(output_json)
            with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

            return scores

        else:
            next_tokens_id = torch.argmax(scores, dim=-1)
            next_token = self.main_model_tokenizer.convert_ids_to_tokens(next_tokens_id)
            self.ensemble_model_output_ids_queue.put(next_token)
            return scores
