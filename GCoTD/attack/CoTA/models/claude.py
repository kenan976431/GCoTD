import torch
from .model import ModelHandler
import anthropic


class ClaudeHandler(ModelHandler):
    def __init__(self, model_name='claude-3-5-sonnet-20241022', api_key='xxxxxxxxxx'):

        self.client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.model=model_name

        return

    def response(self, prompt):
        self.client = self.client.messages.create(
            model=self.model,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
        output = self.completion.choices[0].content

        return output