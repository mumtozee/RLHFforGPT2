from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PromptRequest(_message.Message):
    __slots__ = ("prompt", "generation_params")
    class GenerationParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    GENERATION_PARAMS_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    generation_params: _containers.ScalarMap[str, str]
    def __init__(self, prompt: _Optional[str] = ..., generation_params: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PromptResponse(_message.Message):
    __slots__ = ("generated_text", "usage_information")
    class UsageInformationEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    GENERATED_TEXT_FIELD_NUMBER: _ClassVar[int]
    USAGE_INFORMATION_FIELD_NUMBER: _ClassVar[int]
    generated_text: str
    usage_information: _containers.ScalarMap[str, int]
    def __init__(self, generated_text: _Optional[str] = ..., usage_information: _Optional[_Mapping[str, int]] = ...) -> None: ...
