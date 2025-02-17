from .MATH import parse_pred_ans_json as MATH_parse_pred_ans_json
from .csqa import parse_pred_ans_json as CSQA_parse_pred_ans_json
from .letter import parse_pred_ans_json as Letter_parse_pred_ans_json
from .mmlu import parse_pred_ans_json as MMLU_parse_pred_ans_json

eval_handlers = {
    "MATH": MATH_parse_pred_ans_json,
    "csqa": CSQA_parse_pred_ans_json,
    "letter": Letter_parse_pred_ans_json,
    "mmlu": MMLU_parse_pred_ans_json
}