import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import comfy.model_management as mm
import folder_paths
import torch
import torchaudio


BUNDLED_VOXCPM_ROOT = Path(__file__).resolve().parent
VOXCPM_PRIMARY_MODELS_ROOT = Path(folder_paths.models_dir) / "VoxCPM2"
VOXCPM_FALLBACK_MODELS_ROOT = Path(folder_paths.models_dir) / "VoxCPM"
VOXCPM_MODEL_ROOTS = [
    VOXCPM_PRIMARY_MODELS_ROOT,
    VOXCPM_FALLBACK_MODELS_ROOT,
]

folder_paths.add_model_folder_path("VoxCPM2", str(VOXCPM_PRIMARY_MODELS_ROOT), is_default=True)
folder_paths.add_model_folder_path("VoxCPM", str(VOXCPM_PRIMARY_MODELS_ROOT), is_default=True)
folder_paths.add_model_folder_path("VoxCPM", str(VOXCPM_FALLBACK_MODELS_ROOT), is_default=False)

_MODEL_CACHE: dict[tuple[str, bool, bool], dict[str, Any]] = {}
_LAST_CACHE_KEY: tuple[str, bool, bool] | None = None


GENERATION_INPUTS = {
    "cfg_value": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
    "inference_timesteps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
    "max_len": ("INT", {"default": 4096, "min": 64, "max": 65536, "step": 64}),
    "normalize": ("BOOLEAN", {"default": False}),
    "denoise": ("BOOLEAN", {"default": False}),
}


def _import_voxcpm():
    bundled_path = str(BUNDLED_VOXCPM_ROOT)
    if bundled_path not in sys.path:
        sys.path.insert(0, bundled_path)

    try:
        import voxcpm  # type: ignore

        return voxcpm
    except ImportError as original_error:
        raise ImportError(
            "Cannot import bundled voxcpm source from "
            f"{BUNDLED_VOXCPM_ROOT}."
        ) from original_error


def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "pytorch_model.pt",
    )
    return any((path / name).exists() for name in weight_names)


def _list_model_options() -> list[str]:
    options: list[str] = []
    seen: set[str] = set()

    for root in VOXCPM_MODEL_ROOTS:
        root_label = root.name
        if _is_model_dir(root) and root_label not in seen:
            options.append(root_label)
            seen.add(root_label)

        if root.is_dir():
            for child in sorted(root.iterdir()):
                option = f"{root_label}/{child.name}"
                if _is_model_dir(child) and option not in seen:
                    options.append(option)
                    seen.add(option)

    return options or ["VoxCPM2"]


def _resolve_model_path(model_name: str) -> Path:
    if not model_name:
        for root in VOXCPM_MODEL_ROOTS:
            if _is_model_dir(root):
                return root
        tried_roots = ", ".join(str(root) for root in VOXCPM_MODEL_ROOTS)
        raise FileNotFoundError(f"VoxCPM model directory not found. Tried: {tried_roots}")

    candidate = Path(model_name)
    if candidate.is_absolute() and _is_model_dir(candidate):
        return candidate

    normalized = model_name.strip().strip("/\\")
    search_candidates: list[Path] = []

    if normalized in ("", "VoxCPM", "VoxCPM2"):
        search_candidates.extend(VOXCPM_MODEL_ROOTS)
    else:
        for root in VOXCPM_MODEL_ROOTS:
            search_candidates.append(root / normalized)
            search_candidates.append(root / Path(normalized).name)
        search_candidates.append(Path(folder_paths.models_dir) / normalized)

    unique_candidates: list[Path] = []
    seen = set()
    for path in search_candidates:
        path_str = str(path)
        if path_str not in seen:
            unique_candidates.append(path)
            seen.add(path_str)

    for path in unique_candidates:
        if _is_model_dir(path):
            return path

    tried = ", ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(f"Cannot find a valid VoxCPM model directory. Tried: {tried}")


def _get_or_load_model(model_path: Path, load_denoiser: bool, optimize: bool) -> dict[str, Any]:
    global _LAST_CACHE_KEY

    cache_key = (str(model_path), bool(load_denoiser), bool(optimize))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if _LAST_CACHE_KEY is not None and _LAST_CACHE_KEY != cache_key:
        _MODEL_CACHE.clear()
        mm.soft_empty_cache()

    voxcpm = _import_voxcpm()
    print(f"[ComfyUI-VoxCPM] Loading model from: {model_path}")
    model = voxcpm.VoxCPM.from_pretrained(
        hf_model_id=str(model_path),
        load_denoiser=bool(load_denoiser),
        local_files_only=True,
        optimize=bool(optimize),
    )

    wrapper = {
        "model": model,
        "model_path": str(model_path),
        "load_denoiser": bool(load_denoiser),
        "optimize": bool(optimize),
    }
    _MODEL_CACHE[cache_key] = wrapper
    _LAST_CACHE_KEY = cache_key
    return wrapper


def _audio_to_temp_wav(audio: dict[str, Any] | None, prefix: str) -> str | None:
    if audio is None:
        return None

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.dim() != 2:
        raise ValueError(f"Unsupported audio tensor shape: {tuple(waveform.shape)}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.detach().cpu().to(torch.float32).contiguous()

    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name

    torchaudio.save(temp_path, waveform, sample_rate)
    return temp_path


def _cleanup_temp_paths(paths: list[str]) -> None:
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _compose_text(text: str, control_instruction: str = "") -> str:
    target_text = (text or "").strip()
    if not target_text:
        raise ValueError("text cannot be empty")

    control = (control_instruction or "").strip()
    return f"({control}){target_text}" if control else target_text


def _prepare_output_audio(wav: Any, sample_rate: int) -> dict[str, Any]:
    waveform = torch.as_tensor(wav, dtype=torch.float32).detach().cpu()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Unexpected VoxCPM output shape: {tuple(waveform.shape)}")
    return {
        "waveform": waveform.unsqueeze(0),
        "sample_rate": int(sample_rate),
    }


def _run_generation(
    model_wrapper,
    text,
    control_instruction="",
    reference_audio=None,
    prompt_audio=None,
    prompt_text="",
    cfg_value=2.0,
    inference_timesteps=10,
    max_len=4096,
    normalize=False,
    denoise=False,
):
    current_model = model_wrapper["model"]
    final_text = _compose_text(text, control_instruction)

    prompt_text_clean = (prompt_text or "").strip() or None
    if prompt_audio is None and prompt_text_clean is not None:
        raise ValueError("prompt_text requires prompt_audio")
    if prompt_audio is not None and prompt_text_clean is None:
        raise ValueError("prompt_audio requires prompt_text")

    temp_paths: list[str] = []
    try:
        prompt_wav_path = _audio_to_temp_wav(prompt_audio, "voxcpm_prompt_")
        reference_wav_path = _audio_to_temp_wav(reference_audio, "voxcpm_reference_")

        if prompt_wav_path is not None:
            temp_paths.append(prompt_wav_path)
        if reference_wav_path is not None:
            temp_paths.append(reference_wav_path)

        wav = current_model.generate(
            text=final_text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text_clean,
            reference_wav_path=reference_wav_path,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            max_len=int(max_len),
            normalize=bool(normalize),
            denoise=bool(denoise),
        )

        audio = _prepare_output_audio(wav, current_model.tts_model.sample_rate)
        return (audio,)
    finally:
        _cleanup_temp_paths(temp_paths)


class VoxCPMModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_model_options(), {"default": "VoxCPM2"}),
                "load_denoiser": ("BOOLEAN", {"default": False}),
                "optimize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VOXCPM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/VoxCPM"

    def load_model(self, model_name, load_denoiser, optimize):
        model_path = _resolve_model_path(model_name)
        return (_get_or_load_model(model_path, load_denoiser, optimize),)


class VoxCPMMultilingualTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello. 你好。Bonjour.",
                    },
                ),
                **GENERATION_INPUTS,
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/VoxCPM"

    def generate(self, model, text, cfg_value, inference_timesteps, max_len, normalize, denoise):
        return _run_generation(
            model_wrapper=model,
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            normalize=normalize,
            denoise=denoise,
        )


class VoxCPMVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你好，欢迎使用 VoxCPM2。",
                    },
                ),
                "voice_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "年轻女性，声音温柔甜美，语速适中",
                    },
                ),
                **GENERATION_INPUTS,
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/VoxCPM"

    def generate(self, model, text, voice_description, cfg_value, inference_timesteps, max_len, normalize, denoise):
        return _run_generation(
            model_wrapper=model,
            text=text,
            control_instruction=voice_description,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            normalize=normalize,
            denoise=denoise,
        )


class VoxCPMControllableCloning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "reference_audio": ("AUDIO",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "这是带风格控制的克隆语音演示。",
                    },
                ),
                "style_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                **GENERATION_INPUTS,
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/VoxCPM"

    def generate(
        self,
        model,
        reference_audio,
        text,
        style_instruction,
        cfg_value,
        inference_timesteps,
        max_len,
        normalize,
        denoise,
    ):
        return _run_generation(
            model_wrapper=model,
            text=text,
            control_instruction=style_instruction,
            reference_audio=reference_audio,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            normalize=normalize,
            denoise=denoise,
        )


class VoxCPMUltimateCloning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "prompt_audio": ("AUDIO",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "这是极致克隆演示。",
                    },
                ),
                **GENERATION_INPUTS,
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "参考音频的文本转录。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/VoxCPM"

    def generate(
        self,
        model,
        prompt_audio,
        prompt_text,
        text,
        cfg_value,
        inference_timesteps,
        max_len,
        normalize,
        denoise,
        reference_audio=None,
    ):
        return _run_generation(
            model_wrapper=model,
            text=text,
            reference_audio=reference_audio,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            normalize=normalize,
            denoise=denoise,
        )


class VoxCPMGenerateAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VOXCPM_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你好，这是一个 VoxCPM ComfyUI 节点测试。",
                    },
                ),
                "control_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                **GENERATION_INPUTS,
            },
            "optional": {
                "prompt_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",),
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/VoxCPM/Advanced"

    def generate(
        self,
        model,
        text,
        control_instruction,
        cfg_value,
        inference_timesteps,
        max_len,
        normalize,
        denoise,
        reference_audio=None,
        prompt_audio=None,
        prompt_text="",
    ):
        return _run_generation(
            model_wrapper=model,
            text=text,
            control_instruction=control_instruction,
            reference_audio=reference_audio,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            normalize=normalize,
            denoise=denoise,
        )


NODE_CLASS_MAPPINGS = {
    "VoxCPMModelLoader": VoxCPMModelLoader,
    "VoxCPMMultilingualTTS": VoxCPMMultilingualTTS,
    "VoxCPMVoiceDesign": VoxCPMVoiceDesign,
    "VoxCPMControllableCloning": VoxCPMControllableCloning,
    "VoxCPMUltimateCloning": VoxCPMUltimateCloning,
    "VoxCPMGenerateAudio": VoxCPMGenerateAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxCPMModelLoader": "Load VoxCPM Model",
    "VoxCPMMultilingualTTS": "VoxCPM Multilingual TTS",
    "VoxCPMVoiceDesign": "VoxCPM Voice Design",
    "VoxCPMControllableCloning": "VoxCPM Controllable Cloning",
    "VoxCPMUltimateCloning": "VoxCPM Ultimate Cloning",
    "VoxCPMGenerateAudio": "VoxCPM Generate Audio Advanced",
}
