autogpt/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── ai_config.py
│   ├── config.py
│   └── singleton.py
└── requirements.txt
__all__ = ['AIConfig', 'Config', 'check_openai_api_key', 'AbstractSingleton', 'Singleton']
from .ai_config import AIConfig
from .config import Config, check_openai_api_key
from .singleton import AbstractSingleton, Singleton

__all__ = ['AIConfig', 'Config', 'check_openai_api_key', 'AbstractSingleton', 'Singleton']
numpy==1.21.2
scipy==1.7.1
openai==0.27.0
def some_function():
    pass  # TODO: implement this function

class SomeClass:
    pass  # TODO: implement this class
