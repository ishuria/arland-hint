import json
from random import random
import numpy as np
import torch
import os
from util.file_util import read_file_content_as_array, index_to_path
from util.db_util import get_it_item_index_id_mapping
from util.config_util import DATA_FOLDER


class VectorCalculator:

    def __init__(self, model_vector):
        self.model_vector = model_vector
        sorted_model_vector = sorted(self.model_vector)
        self.sorted_index = {}
        self.indexed_model_vector = []
        for i in range(len(sorted_model_vector)):
            self.sorted_index[sorted_model_vector[i]] = i
        for i in range(len(self.model_vector)):
            self.indexed_model_vector.append(self.sorted_index[self.model_vector[i]])

    def print_model_vector(self):
        print(self.model_vector)
        print(self.sorted_index)
        print(self.indexed_model_vector)


if __name__ == '__main__':
    with open('./kancd/student_knowledge_vector.json', 'r') as f:
        lines = "".join(f.readlines())
        student_knowledge_vector = json.loads(lines)

    with open('./class.txt', 'r') as f:
        knowledges = [line.rstrip('\n') for line in f]

    # print(knowledges)
    
    it_item_index_id_mapping = get_it_item_index_id_mapping()
    # print(it_item_index_id_mapping)
    student_knowledge_vector_calculator = VectorCalculator(student_knowledge_vector)

    for i in range(len(it_item_index_id_mapping)):
        index = it_item_index_id_mapping[i + 1]
        knowledge_list = read_file_content_as_array(index, ['knowledge_vector'], True)
        indexed_knowledge_list = VectorCalculator(knowledge_list)
        print(indexed_knowledge_list.indexed_model_vector)
        x = (np.subtract(indexed_knowledge_list.indexed_model_vector,student_knowledge_vector_calculator.indexed_model_vector))
        topx = torch.topk(torch.Tensor(x), 3)
        # print(topx.values.item())
        top_knowledges = []
        for knowledge_index in topx.values:
            print(int(knowledge_index.item()))
            print(knowledges[int(knowledge_index.item())])
            top_knowledges.append(knowledges[int(knowledge_index.item())])

        print("writing to file: ", DATA_FOLDER + index_to_path(index) + os.sep + 'top_knowledge.txt')
        with open(DATA_FOLDER + index_to_path(index) + os.sep + 'top_knowledge.txt', 'w') as eval_file:
            eval_file.write('\n'.join(top_knowledges))


    # random_vector = []
    # for i in range(len(student_knowledge_vector)):
    #     random_vector.append(random())
    # student_knowledge_vector_calculator = VectorCalculator(student_knowledge_vector)
    # # student_knowledge_vector_calculator.print_model_vector()
    # random_vector_calculator = VectorCalculator(random_vector)
    # # random_vector_calculator.print_model_vector()
    # x = (np.subtract(student_knowledge_vector_calculator.indexed_model_vector,
    #                  random_vector_calculator.indexed_model_vector))
    # topx = torch.topk(torch.Tensor(x), 3)
    # print(topx)
