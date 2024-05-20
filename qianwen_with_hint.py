#sk-a6b309c1f56c420a87b74034c63d8ae5
import json
from http import HTTPStatus
import dashscope
import numpy as np

from util.db_util import get_it_item_index_id_item_id_mapping, save_llm_answer, get_llm_hint
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


    for i in range(len(it_item_index_id_mapping)):
        print("index: ", i + 1, " of total: ", len(it_item_index_id_mapping))
        index = it_item_index_id_mapping[i + 1]
        item_id = id_item_id_map[index]
        content = read_file_content_as_string(index, ['content'], True)
        new_hint = read_file_content_as_string(index, ['new_hint'], True)
        answer = read_file_content_as_string(index, ['answer'], True)

        if str(item_id) not in item_features:
            continue

        hint = get_llm_hint("qwen_turbo", item_id)
        print(hint)
        if hint is None or hint == '':
            continue
        min_knowledge_cd = [x-float(y) for (x,y) in zip(knowledge_cd, item_features[elements[0]])]
        min_knowledge_index = np.argmin(min_knowledge_cd)
        min_knowledge = knowledge_id_name_dict[str(min_knowledge_index + 1)]

        request = '<回答选择题>:' + content + '<提示>:' + hint + '<回答请用"答案是A"的格式><不要依赖提示中的答案，自行判断>'
        # request = '<回答选择题>:' + content + '<回答请用"答案是A"的格式>'
        response = call_with_prompt(request)
        llm_original_answer = json.dumps(response)
        text = response["output"]["text"]

        extracted_answer = extract_answer_from_str_qianwen(text)
        if extracted_answer is None:
            extracted_answer = text
            if len(extracted_answer) > 100:
                extracted_answer = extracted_answer[:100]
        bot_answer = []
        if 'A' in extracted_answer:
            bot_answer.append('A')
        if 'B' in extracted_answer:
            bot_answer.append('B')
        if 'C' in extracted_answer:
            bot_answer.append('C')
        if 'D' in extracted_answer:
            bot_answer.append('D')
        print('bot answer: ', bot_answer)
        llm_answer_json = json.dumps(bot_answer)
        true_answer = []
        if 'A' in answer:
            true_answer.append('A')
        if 'B' in answer:
            true_answer.append('B')
        if 'C' in answer:
            true_answer.append('C')
        if 'D' in answer:
            true_answer.append('D')
        print('true answer: ', true_answer)
        true_answer_json = json.dumps(true_answer)

        max_length = len(bot_answer)
        if len(true_answer) > len(bot_answer):
            max_length = len(true_answer)

        match_count = 0
        for ta in true_answer:
            for ba in bot_answer:
                if ba == ta:
                    match_count += 1
                    break
        if len(true_answer) == 0:
            continue

        score = round(float(match_count) / float(len(true_answer)), 2)
        if len(bot_answer) > len(true_answer):
            score = 0.0

        print('score: ', score)

        total_score += score
        print("current average score: ", total_score / (i + 1))

        save_llm_answer('qwen_turbo_hint_3',
                        llm_original_answer,
                        answer,
                        llm_answer_json,
                        true_answer_json,
                        score,
                        item_id)
