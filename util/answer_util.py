# coding=utf8
# the above tag defines encoding for this document and is for Python 2.x compatibility

import re


def extract_answer_from_str(answer_str: str):
    regex = r"正确答案为.*?([A-Z]*)|答.*?([A-Z]+)|正确答案是.*?([A-Z]+)"
    matches = re.finditer(regex, test_str, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        # print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1
            if match.group(groupNum) is None:
                continue
            return match.group(groupNum)
            # print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))
    return None

if __name__ == '__main__':
    test_str = ("故正确答案为AB\n"
        "答：B\n"
        "正确答案是 A")
    print(extract_answer_from_str(test_str))