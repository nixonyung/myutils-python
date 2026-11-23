import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self, overload

try:
    import ollama
    from numind.nuextract_utils import (
        convert_json_schema_to_nuextract_template,
    )
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "Missing dependencies for myutils.ollama. Please install with extra flag: 'myutils[ollama]'"
    ) from e


def ensure_model_exists(model: str):
    if model in {m["model"] for m in ollama.list().models}:
        return

    print(f"model {model} missing from ollama, pulling...")
    ollama.pull(model)


class OllamaRunner[Format: type[BaseModel] | None](ABC):
    """
    Usage 1:

    ```
    runner = _OllamaRunner(...)

    for content in runner:  # e.g. wrap with tqdm to show progress bar
        print(content)

    result = runner.result
    ```

    Usage 2:

    ```
    result = _OllamaRunner(...).collect()
    ```
    """

    def __init__(
        self,
        model: str,
        format: Format,
        think: bool | None,
        options: ollama.Options,
    ):
        self._model = model
        self._output_cls = format
        self._think = think
        self._options = options

        self._result: str | None = None

        ensure_model_exists(self._model)

    @property
    @abstractmethod
    def messages(self) -> list[ollama.Message]:
        pass

    def __iter__(self):
        stream = ollama.chat(
            model=self._model,
            messages=self.messages,
            format=self._output_cls.model_json_schema() if self._output_cls is not None else None,
            think=self._think,
            options=self._options,
            stream=True,
        )

        contents: list[str] = []
        for chunk in stream:
            content = chunk.message.content or ""
            contents.append(content)

            yield content

        self._result = "".join(contents)

    class _ResultDescriptor:
        @overload
        def __get__(self, instance: None, owner: type) -> Self: ...

        @overload
        def __get__(self, instance: "OllamaRunner[None]", owner: type) -> str: ...

        @overload
        def __get__[M: BaseModel](self, instance: "OllamaRunner[type[M]]", owner: type) -> M: ...

        def __get__(
            self,
            instance: "None | OllamaRunner[Any]",
            owner: type,
        ) -> Self | str | BaseModel:
            if instance is None:
                return self

            if instance._result is None:
                raise RuntimeError("The stream has not finished. Consume the runner first.")

            return (
                instance._output_cls.model_validate_json(instance._result)
                if issubclass(instance._output_cls, BaseModel)
                else instance._result
            )

    result: _ResultDescriptor = _ResultDescriptor()

    @overload
    def collect(self: "OllamaRunner[None]") -> str: ...

    @overload
    def collect[M: BaseModel](self: "OllamaRunner[type[M]]") -> M: ...

    def collect(self: "OllamaRunner[Any]") -> Any:
        for _ in self:
            pass

        return self.result


class NuExtract3Runner[Format: type[BaseModel]](OllamaRunner[Format]):
    def __init__(
        self,
        image_path: Path,
        format: Format,
        *,
        model="numind/nuextract3:Q6_K",
        think=False,
        temperature=0.0,
        num_ctx=8192,
        num_predict=-1,
    ):
        super().__init__(
            model,
            format,
            think,
            ollama.Options(
                temperature=temperature,
                num_ctx=num_ctx,
                num_predict=num_predict,
            ),
        )
        self._image_path = image_path

    @property
    def messages(self):
        return [
            ollama.Message(
                role="template",
                content=json.dumps(
                    convert_json_schema_to_nuextract_template(self._output_cls.model_json_schema())
                ),
            ),
            ollama.Message(
                role="user",
                images=[ollama.Image(value=self._image_path)],
            ),
        ]


__all__ = [
    "NuExtract3Runner",
    "ensure_model_exists",
]
