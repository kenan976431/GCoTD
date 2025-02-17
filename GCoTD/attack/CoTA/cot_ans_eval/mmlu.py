import re
import json
from collections import Counter


def shift_answer(ans, shift, mode=None):
    # Convert to number
    # if not ood:
    ans_list = ['A', 'B', 'C', 'D']
    ans_ind = ans_list.index(ans)
    ans_ind_shifted = (ans_ind + shift) % len(ans_list)
    ans_shifted = ans_list[ans_ind_shifted]

    return ans_shifted


def test_answer(pred_str, ans_str, shift):

    pred_str = pred_str.split('answer is ')[-1]
    pred = re.findall(r'A|B|C|D', pred_str)
    gold = re.findall(r'A|B|C|D', ans_str)[-1]
    if len(pred) == 1:
        return pred[0] == shift_answer(gold, shift)
    else:
        return False


def step_exists(pred_str, shift):

    sentences = pred_str.split('\n')
    choices = []
    for i in range(len(sentences)):
        sentence = sentences[len(sentences) - 1 - i]
        if sentence.find('next letter of') > 0:
            choices = re.findall(r'A|B|C|D', sentence)
            break
    if len(choices) < 2:
        return False
    
    return choices[1] == shift_answer(choices[0], shift)


def parse_pred_ans_json(filename, shift=0, check_step=False):
    with open(filename, 'r') as f:
        res = json.load(f)

    num_q = len(res)
    acc = 0

    for i in range(len(res)):
        q = res[i]['Question']
        am_list = res[i]['Model Answer']
        a = res[i]['Ref Answer']

        # handle PaLM Failure case
        if 'palm' in filename:
            skip_q = False
            for am in am_list:
                if "PaLM failed" in am:
                    skip_q = True
                    break
            if skip_q:
                num_q -= 1
                continue
        
        sc = len(am_list)

        if sc == 1: # CoT
            am = am_list[0]
            if check_step:
                if test_answer(am, a, shift) or step_exists(am, shift):
                    acc += 1
            else:
                if test_answer(am, a, shift):
                    acc += 1

        else: # self-consistency
            add_acc = False

            # 1. check majority result
            pred_list = []
            for am in am_list:
                pred = am.split('answer is ')[-1]
                pred = re.findall(r'A|B|C|D', pred)
                if len(pred) >= 1:
                    pred_list.append(pred[0])
                else:
                    continue
            if Counter(pred_list).most_common() != []:
                majority = Counter(pred_list).most_common(1)[0][0]
                gold = re.findall(r'A|B|C|D', a)[-1]
                
                if majority in ["A", "B", "C", "D"]:
                    add_acc = majority == shift_answer(gold, shift)
                    
            # 2. check step
            if check_step:
                count = 0
                for am in am_list:
                    if step_exists(am, shift):
                        count += 1
                if count >= sc/2:
                    add_acc = True
                
            if add_acc:
                acc += 1

    print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
    return num_q, acc, float(acc / num_q)