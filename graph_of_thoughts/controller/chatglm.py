# author: Jayce Ning

import os
import random
import time
from typing import List, Dict, Union
import requests
import json

from .abstract_language_model import AbstractLanguageModel


class ChatGLM(AbstractLanguageModel):
    """
    The ChatGLM class handles interactions with the ChatGLM using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "chatglm", cache: bool = False
    ) -> None:
        """
        Initialize the ChatGLM instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'chatglm'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used for chatglm, i.e. chatglm2-6b, etc.
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        # self.temperature: float = self.config["temperature"]
        self.url: str = self.config["url"]
        # The maximum number of tokens to generate in the chat completion.
        # self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at (it will not generate the stop sequence).
        # self.stop: Union[str, List[str]] = self.config["stop"]


    def query(self, query: str, num_responses: int = 1) -> Dict:
        """
        Query the ChatGLM model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the ChatGLM model.
        :rtype: Dict
        """
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]

        if num_responses == 1:
            response = self.chat([{"role": "user", "content": query}], num_responses)
        else:
            response = []
            next_try = 1  # chatglm interface is set to single query
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    res = self.chat([{"role": "user", "content": query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"Error in chatglm: {e}, trying again with {next_try} samples"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        if self.cache:
            self.respone_cache[query] = response
        return response

    def chat(self, messages: List[Dict], num_responses: int = 1) -> Dict:
        """
        Send chat messages to the ChatGLM Model and retrieves the model's response.
        
        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The ChatGLM model's response.
        :rtype: Dict
        """

        # 定义请求的URL和数据 "http://127.0.0.1:8000"
        url = self.url
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "prompt": messages[0]["content"],
            "history": []
        }

        # 将Python对象转换为JSON字符串
        json_data = json.dumps(data)

        # 发送POST请求
        response = requests.post(url, headers=headers, data=json_data).text
        response = json.loads(response)
    
        self.prompt_tokens += response["prompt_token"]
        self.completion_tokens += response["response_token"]
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            f"This is the response from chatglm: {response}"
            f"\nThis is the cost of the response: {self.cost}"
        )
        return response

    def get_response_texts(self, query_response: Union[List[Dict], Dict]) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the ChatGLM2 model.
        :type query_response: Union[List[Dict], Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        if isinstance(query_response, Dict):
            query_response = [query_response]
        return [
            response["response"]
            for response in query_response
        ]
