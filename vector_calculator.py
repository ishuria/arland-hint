import json
from random import random
import numpy as np
import torch


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
    # print(student_knowledge_vector)
    random_vector = []
    for i in range(len(student_knowledge_vector)):
        random_vector.append(random())
    student_knowledge_vector_calculator = VectorCalculator(student_knowledge_vector)
    # student_knowledge_vector_calculator.print_model_vector()
    random_vector_calculator = VectorCalculator(random_vector)
    # random_vector_calculator.print_model_vector()
    x = (np.subtract(student_knowledge_vector_calculator.indexed_model_vector,
                     random_vector_calculator.indexed_model_vector))
    topx = torch.topk(torch.Tensor(x), 3)
    print(topx)
