import torch
from .model import ModelHandler
from openai import OpenAI


class DeepseekHandlerV3(ModelHandler):
    def __init__(self, model_name='deepseek-chat', api_key='xxxxxxxxxx'):

        self.client = OpenAI(
            base_url="https://api.deepseek.com", 
            api_key=api_key,
        )
        self.model=model_name

        return

    def response(self, prompt):
        self.completion = self.client.chat.completions.create(
            model=self.model,
            messages=[    
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
        output = self.completion.choices[0].message.content

        return output