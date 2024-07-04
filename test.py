import torch,json
PROMPT = """Answer with the option's letter from the given choices directly."""
yuan_data='/data/ouyangxc/github/CoT-V/datasets/aok/custom/aok_cot_eval.jsonl'
yuan_data = [json.loads(q) for q in open(yuan_data, "r")]
stage_1_answer='/data/ouyangxc/github/imp/playground/data/eval/aok/answers/eval/cot_73/merge.jsonl'
stage_1={}
answers_file_path='/data/ouyangxc/github/imp/playground/data/eval/aok/qok_base_input.jsonl'
for q in open(stage_1_answer, "r"):
    line=json.loads(q)
    stage_1[line['question_id']]=line['text']
for x in yuan_data:
    inputs=PROMPT
    input_list=x['text'].split('\n')
    question,opetions=input_list[1],input_list[2]
    input_list[0]=inputs
    final='\n'.join([input_list[0],input_list[1],input_list[2]])+'\nAnswer: '
    x['text']=final
    with open(answers_file_path, "a") as f:
        f.write(json.dumps(x) + "\n")
    f.close()