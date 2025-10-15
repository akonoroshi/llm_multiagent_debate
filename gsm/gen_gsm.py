import json
import numpy as np
import random
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion.content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def write_jsonl(path: str, data):
    with open(path, 'w') as fh:
        for item in data:
            fh.write(json.dumps(item, indent=4) + '\n')

if __name__ == "__main__":
    agents = 3
    rounds = 2
    random.seed(0)
    model = ["gpt-3.5-turbo-0301", "gpt-oss"][1]
    if "gpt" in model and not "gpt-oss" in model:
        llm = ChatOpenAI(model_name=model, temperature=0)
    else:
        llm = ChatOllama(model=model)

    generated_description = []

    questions = read_jsonl("/ix1/dlitman/yua17/grade-school-math/grade_school_math/data/train.jsonl")
    random.shuffle(questions)

    for data in questions[:100]:
        output = {}
        question = data['question']
        answer = data['answer']
        output['question'] = question
        output['answer'] = answer
        output['turns'] = []

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                completion = llm.invoke(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                output['turns'].append({
                    "agent": i,
                    "round": round,
                    "response": assistant_message
                })

        generated_description.append(output)

    write_jsonl(f"gsm_multiagent_{model}_{agents}_{rounds}.jsonl", generated_description)
