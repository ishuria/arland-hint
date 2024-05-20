#sk-a6b309c1f56c420a87b74034c63d8ae5
import json
from http import HTTPStatus
import dashscope
import numpy as np

from util.db_util import get_it_item_index_id_item_id_mapping, save_llm_answer, save_llm_hint, get_llm_hint
from util.file_util import read_file_content_as_string, read_file_content
from util.answer_util import extract_answer_from_str_qianwen


def call_with_prompt(prompt):
    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_turbo,
        prompt=prompt
    )
    # 如果调用成功，则打印模型的输出
    if response.status_code == HTTPStatus.OK:
        # print(response)
        return response
    # 如果调用失败，则打印出错误码与失败信息
    else:
        print(response.code)
        print(response.message)


if __name__ == '__main__':
    it_item_index_id_mapping, id_item_id_map = get_it_item_index_id_item_id_mapping()
    total_score = 0

    knowledge_cd_str = ''
    with open('./answer/cd.txt', 'r', encoding="utf-8") as f:
        for line in f:
            knowledge_cd_str += line.rstrip('\n')
    knowledge_cd = json.loads(knowledge_cd_str)

    knowledge_id_name_dict = {}
    with open('./answer/item.knowledge.name', 'r', encoding="utf-8") as f:
        for line in f:
            elements = line.rstrip('\n').split('\t')
            knowledge_id_name_dict[elements[1]] = elements[0]

    item_features = {}
    with open('./answer/item.content', 'r', encoding="utf-8") as f:
        for line in f:
            feature = []
            elements = line.rstrip('\n').split('\t')
            for i in range(len(elements)):
                if i == 0:
                    continue
                if i == len(elements) - 1:
                    continue
                feature.append(elements[i])
            item_features[elements[0]] = feature

    max_cd = max(knowledge_cd)
    min_cd = min(knowledge_cd)

    for i in range(len(knowledge_cd)):
        knowledge_cd[i] = (knowledge_cd[i] - min_cd) / (max_cd - min_cd)

    min_knowledge_index = np.argmin(knowledge_cd)
    min_knowledge = knowledge_id_name_dict[str(min_knowledge_index + 1)]
    for i in range(len(it_item_index_id_mapping)):
        print("index: ", i + 1, " of total: ", len(it_item_index_id_mapping))
        index = it_item_index_id_mapping[i + 1]
        item_id = id_item_id_map[index]
        content = read_file_content_as_string(index, ['content'], True)
        new_hint = read_file_content_as_string(index, ['new_hint'], True)
        answer = read_file_content_as_string(index, ['answer'], True)

        if str(item_id) not in item_features:
            continue

        hint = get_llm_hint("qwen_turbo_no_answer", item_id)
        print(hint)
        if len(hint) != 0:
            continue

        min_knowledge_cd = [x-float(y) for (x,y) in zip(knowledge_cd, item_features[elements[0]])]
        min_knowledge_index = np.argmin(min_knowledge_cd)
        min_knowledge = knowledge_id_name_dict[str(min_knowledge_index + 1)]

        print(content)
        print(min_knowledge)
        print(answer)

        request = '<结合问题与答案与知识点生成提示信息>\n<提示信息不能包含答案>\n<问题>:' + content + '\n<知识点>:' + str(min_knowledge) + '\n<答案>:' + answer
        response = call_with_prompt(request)
        llm_original_answer = json.dumps(response)
        hint = response["output"]["text"]

        save_llm_hint('qwen_turbo_no_answer',
                        hint,
                        item_id)
