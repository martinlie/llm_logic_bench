import os, re, json, uuid
from dotenv import load_dotenv
_ = load_dotenv()
#import pandas as pd
#import numpy as np
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional
import json

MAX_STEPS = 20   # prevent infinite loops

class BQAModel(BaseModel):
    reasoning: str
    answer: str
    is_answer_yes: bool

def main():
      key = os.getenv("SCW_OPENAI_SECRET_KEY", "")
      client = OpenAI(
            base_url = "https://api.scaleway.ai/v1",
            api_key=key
      )

      filename = Path("./LogicBench") / "data" / "LogicBench-Eval" / "BQA" / "first_order_logic" / "bidirectional_dilemma" / "data_instances.json"
      #ai_model = "qwen3-235b-a22b-instruct-2507" # 65%
      ai_model = "gpt-oss-120b" # 71/80= 88.75%
      #ai_model = "gemma-3-27b-it" # 65%
      #ai_model = "mistral-small-3.2-24b-instruct-2506" # 35/80=43.75%
      #ai_model = "llama-3.3-70b-instruct" # 67/80=83.75%
      #ai_model = "deepseek-r1-distill-llama-70b" # 21/42=50%

      with open(filename, "r") as f:
            data = json.load(f)

      no_of_samples = 0
      correct_samples = 0

      for sample in data['samples']:
            id = sample['id']
            context = sample['context']
            print(f"({id}): {context}")
            for qa_pair in sample['qa_pairs']:
                  no_of_samples += 1

                  print(f"Question: {qa_pair['question']}")
                  print(f"Answer: {qa_pair['answer']}")

                  result = client.chat.completions.parse(
                        model=ai_model,
                        messages=[
                              { "role": "system", "content": context },
                              { "role": "user", "content": qa_pair['question'] },
                        ],
                        #max_tokens=16384,
                        #temperature=0.7,
                        #top_p=0.8,
                        #presence_penalty=0,
                        response_format=BQAModel #{ "type": "text" }
                  )
                  llm_answer = result.choices[0].message.parsed
                  if llm_answer is None:
                        print(result.choices[0].finish_reason)
                        return
                  is_answer_yes = qa_pair['answer'] == 'yes'
                  isCorrect = (is_answer_yes == llm_answer.is_answer_yes) # similar conclusion

                  print(f"AnswerYES={is_answer_yes} LLM answerYES={llm_answer.is_answer_yes} similar/correct?={isCorrect}")
                  print(f"{llm_answer.answer}")
                  print(f"{llm_answer.reasoning}")
                  
                  if isCorrect:
                        correct_samples += 1

                  print(f"------ Samples {no_of_samples}, correct# {correct_samples}, correct {(correct_samples/no_of_samples) * 100:.2f}%")
      
if __name__ in {'__main__', '__mp_main__'}:
      main()