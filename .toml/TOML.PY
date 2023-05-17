# pyproject.toml 파일의 내용을 파이썬 딕셔너리로 표현
config = {
    "build-system": {
        "requires": ["hatchling"],
        "build-backend": "hatchling.build",
    },
    "project": {
        "name": "agpt",
        "version": "0.3.0",
        "authors": [
            {"name": "Torantulino", "email": "support@agpt.co"},
        ],
        "readme": "README.md",
        "requires-python": ">=3.10",
        "classifiers": [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        "description": "An open-source attempt to make GPT-4 autonomous",
        "urls": {
            "Homepage": "https://github.com/rsrrrrrr/Auto-GPT",
            "Bug Tracker": "https://github.com/rsrrrrrr/Auto-GPT",
        },
    },
    "tool": {
        "black": {
            "line-length": 88,
            "target-version": ['py310'],
            "include": '\.pyi?$',
            "packages": ["autogpt"],
            "extend-exclude": '.+/(dist|.venv|venv|build)/.+',
        },
        "isort": {
            "profile": "black",
            "multi_line_output": 3,
            "include_trailing_comma": True,
            "force_grid_wrap": 0,
            "use_parentheses": True,
            "ensure_newline_before_comments": True,
            "line_length": 88,
            "sections": [
                "FUTURE",
                "STDLIB",
                "THIRDPARTY",
                "FIRSTPARTY",
                "LOCALFOLDER",
            ],
            "skip": '''
                .tox
                __pycache__
                *.pyc
                .env
                venv*/*
                .venv/*
                reports/*
                dist/*
            ''',
        },
    },
}

# toml 라이브러리를 사용하여 파이썬 딕셔너리를 toml 형식의 문자열로 변환
import toml

toml_string = toml.dumps(config)

# toml 형식의 문자열을 pyproject.toml 파일에 쓰기
with open("pyproject.toml", "w") as f:
    f.write(toml_string)
