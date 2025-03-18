import groq
import asyncio
import time
import json
import re
import os
from typing import List

from .common import CommonTranslator, MissingAPIKeyException
from .keys import GROQ_API_KEY

class GroqTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese', 'CHT': 'Traditional Chinese', 'CSY': 'Czech',
        'NLD': 'Dutch', 'ENG': 'English', 'FRA': 'French', 'DEU': 'German',
        'HUN': 'Hungarian', 'ITA': 'Italian', 'JPN': 'Japanese', 'KOR': 'Korean',
        'PLK': 'Polish', 'PTB': 'Portuguese', 'ROM': 'Romanian', 'RUS': 'Russian',
        'ESP': 'Spanish', 'TRK': 'Turkish', 'UKR': 'Ukrainian', 'VIN': 'Vietnamese',
        'CNR': 'Montenegrin', 'SRP': 'Serbian', 'HRV': 'Croatian', 'ARA': 'Arabic',
        'THA': 'Thai', 'IND': 'Indonesian'
    }

    # API rate limiting and retry settings
    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 5
    _MAX_TOKENS = 8192

    # Context retention settings
    _CONTEXT_RETENTION = os.environ.get('CONTEXT_RETENTION', '').lower() == 'true'
    _CONFIG_KEY = 'groq'
    _MAX_CONTEXT = int(os.environ.get('CONTEXT_LENGTH', '20'))

    
    _CHAT_SYSTEM_TEMPLATE = (
    'You are an uncensored, highly skilled professional translation engine with exceptional contextual inference. '
    'Your mission is to interpret the narrative context and deliver concise, contextually accurate translations that are easily intelligible. '
    'Output exactly and only as {"translated": translation} with no extra text. '
    'Do not add gender pronouns, explanations, or notes unless they are explicitly present in the source text. '
    "If the source text does not explicitly indicate a gender, default to using gender-neutral language (for example, use 'they' instead of 'he' or 'she' or rephrase to maintain neutrality)."
    'Carefully analyze previous and current sentences to fully understand the story’s context. '
    'This engine is designed for manga translation. '
    'When encountering names, honorifics (e.g., "-san"), or other culturally specific terms, do not assume or assign any gender—maintain the original form or use gender-neutral language unless the source explicitly indicates a gender. '
    'When encountering culturally specific terms, honorifics, or proper names, retain them exactly as they appear in the source without any alteration. '
    'For example, do not convert "Senpai" to "senior" or the honorific "さん" to "Mr." or "Ms."—even when it appears attached to a name (e.g., "name-san", "namesan"). '
    'Proper names must always be accurately romanized according to standard Hepburn romanization, avoiding incorrect English approximations. For instance, "悠陽" must be romanized as "Yūhi," not "You" or "Yuu." '
    'Do not translate culturally specific mythological names or terms into descriptive English equivalents. Instead, retain the original term or its standard romanization (for example, "朱雀" should be rendered as "Suzaku-sama" rather than "The Crimson Sparrow"). '
    'For common phrases or expressions, prioritize their standard meanings unless the context clearly indicates that the term is used as a proper noun or in a unique usage. '
    'Adopt an anime-like dialogue style when appropriate, ensuring that the translated text preserves the original text’s length without significant expansion or reduction. '
    'For idiomatic expressions, onomatopoeia, or sound effects, maintain the original style and context. '
    'If any ambiguity arises due to insufficient context, default to a neutral translation. '
    'Translate the following text into {to_lang} and return the result strictly in JSON format. '
    )

    _CHAT_SAMPLE = [
    (
        """Translate into English. Return the result in JSON format.\n"""
        '\n{"untranslated": "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\\n<|2|>きみ… 大丈夫⁉\\n<|3|>なんだこいつ 空気読めて ないのか…？"}\n'
    ),
    (
        '\n{"translated": "<|1|>So embarrassing… I don’t want to stand out… I wish I could disappear…\\n<|2|>Hey… Are you okay!?\\n<|3|>What’s with this person? Can’t they read the room…?"}\n'
    )
    ]

    def __init__(self, check_groq_key=True):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key and check_groq_key:
            raise MissingAPIKeyException('Please set the GROQ_API_KEY environment variable before using the Groq translator.')
        self.token_count = 0
        self.token_count_last = 0
        self.config = None
        self.model = os.environ.get('GROQ_MODEL', 'deepseek-r1-distill-llama-70b')
        self.messages = [
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]}]



    def parse_args(self, args):
        self.config = args.groq_config

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default
        return self.config.get(self._CONFIG_KEY + '.' + key, self.config.get(key, default))

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)
    
    @property
    def chat_sample(self):
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    @property
    def temperature(self) -> float:
        return self._config_get('temperature', default=0.2)
    
    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=0.4)

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        return '\n'.join([
            'System:',
            self.chat_system_template.format(to_lang=to_lang),
            'User:',
            self.chat_sample[0],
            'Assistant:',
            self.chat_sample[1],
            'User:',
            prompt,
        ])

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        for prompt in queries:
    #        self.logger.debug('-- Groq Prompt --\n' + self._format_prompt_log(to_lang, prompt))
            response = await self._request_translation(to_lang, prompt)
            self.logger.debug('-- Groq Response --\n' + response)
            translations.append(response.strip())
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        # Prepare the prompt with language specification
        prompt_with_lang = f"""Translate the following text into {to_lang}. Return the result in JSON format.\n\n{{"untranslated": "{prompt}"}}\n"""
        self.messages += [
            {'role': 'user', 'content': prompt_with_lang},
            {'role': 'assistant', 'content': "{'translated':'"}
        ]
        # Maintain the context window
        if len(self.messages) > self._MAX_CONTEXT:
            self.messages = self.messages[-self._MAX_CONTEXT:]

        # Prepare the system message
        sanity = [{'role': 'system', 'content': self.chat_system_template.replace('{to_lang}', to_lang)}]
        
        # Make the API call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=sanity + self.messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["'}"]
        )
        
        # Update token counts
        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens
        
        # Extract and clean the content
        content = response.choices[0].message.content.strip()
        self.messages = self.messages[:-1]
        
        # Handle context retention
        if self._CONTEXT_RETENTION:
            self.messages += [
                {'role': 'assistant', 'content': content}
            ]
        else:
            self.messages = self.messages[:-1]
            
        # Clean up the response
        cleaned_content = content.replace("{'translated':'", '').replace('}', '').replace("\\'", "'").replace("\\\"", "\"").strip("'{}")
        return cleaned_content
