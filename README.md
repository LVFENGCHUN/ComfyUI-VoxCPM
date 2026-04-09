# ComfyUI-VoxCPM

ComfyUI custom nodes for VoxCPM2.

This project wraps the official OpenBMB VoxCPM inference code as ComfyUI nodes, with the model loaded from ComfyUI/models/VoxCPM2 by default.

The runtime VoxCPM source code is bundled inside this repository, so the custom node no longer depends on an external ComfyUI/VoxCPM/src checkout.

![ComfyUI-VoxCPM Node Example](https://github.com/user-attachments/assets/cff37898-2e10-48b7-9dca-34964e476a35)

Upstream links:

- Official project: https://github.com/OpenBMB/VoxCPM.git
- Official model: https://huggingface.co/openbmb/VoxCPM2

## Features

Based on the official VoxCPM2 project and model page, this wrapper supports the main VoxCPM2 workflows:

- Multilingual text to speech
- Voice design from natural-language speaker descriptions
- Controllable cloning from short reference audio
- Ultimate cloning with reference audio plus transcript
- 48kHz output from the VoxCPM2 model pipeline
- Context-aware synthesis

Supported languages listed by the official model page:

Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese

Chinese dialects listed by the official model page:

四川话, 粤语, 吴语, 东北话, 河南话, 陕西话, 山东话, 天津话, 闽南话

## Installation

### 1. Install this custom node

Recommended installation command:

    cd ComfyUI/custom_nodes
    git clone https://github.com/starsFriday/ComfyUI-VoxCPM.git

If you prefer manual installation, place the project at:

    ComfyUI/custom_nodes/ComfyUI-VoxCPM

### 2. Install dependencies

Install the Python dependencies from the bundled requirements file:

    pip install -r ComfyUI/custom_nodes/ComfyUI-VoxCPM/requirements.txt

If you prefer using the upstream package directly, you can also install:

    pip install voxcpm

Official upstream requirements mention:

- Python >= 3.10 and < 3.13
- PyTorch >= 2.5.0
- CUDA >= 12.0

### 3. Restart ComfyUI

After installing dependencies and placing the model files, restart ComfyUI.

## Model Download

Official model page:
https://huggingface.co/openbmb/VoxCPM2

Official project page:
https://github.com/OpenBMB/VoxCPM.git

### Recommended model path

Put the model files in:

    ComfyUI/models/VoxCPM2

This wrapper uses VoxCPM2 as the primary model directory.
It also keeps compatibility with the legacy fallback path:

    ComfyUI/models/VoxCPM

### Expected files

At minimum, the local model directory should contain files like:

    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    audiovae.pth

### Download method A: Manual download

Open the Hugging Face page and download the required files into:

    ComfyUI/models/VoxCPM2

### Download method B: Hugging Face CLI

Example:

    huggingface-cli download openbmb/VoxCPM2 --local-dir ./models/VoxCPM2

Run that from the ComfyUI root, or change the local directory to your absolute ComfyUI model path.

### Download method C: Python helper

Example:

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="openbmb/VoxCPM2",
        local_dir="./models/VoxCPM2",
        local_dir_use_symlinks=False,
    )

## Recommended Directory Layout

    ComfyUI/
    ├─ custom_nodes/
    │  └─ ComfyUI-VoxCPM/
    │     ├─ nodes.py
    │     ├─ voxcpm/
    │     ├─ requirements.txt
    │     └─ README.md
    └─ models/
       └─ VoxCPM2/
          ├─ config.json
          ├─ model.safetensors
          ├─ tokenizer.json
          ├─ tokenizer_config.json
          ├─ special_tokens_map.json
          └─ audiovae.pth

## How This Wrapper Works

This custom node ships with a bundled copy of the runtime VoxCPM source code inside the plugin directory.

That means you do not need to keep a separate ComfyUI/VoxCPM/src repository checkout just to run the nodes.

The only external assets still required are the model files under ComfyUI/models/VoxCPM2 and the Python dependencies listed in requirements.txt.

## Included Nodes

### 1. Load VoxCPM Model

Purpose:
Load the VoxCPM2 model once and output a reusable VOXCPM_MODEL object for the generation nodes.

Inputs:
- model_name
- load_denoiser
- optimize

How to use:
1. Add this node first in every workflow.
2. Select the model folder from the dropdown.
3. Keep load_denoiser disabled unless you need denoise support.
4. Keep optimize disabled if you want safer first-run debugging.
5. Connect the model output to any VoxCPM generation node.

When to use load_denoiser:
- Enable it if you plan to use denoise on noisy reference audio.
- Leave it off if you only need plain text to speech and want fewer dependencies.

When to use optimize:
- Enable it for long-term repeated inference.
- Leave it off for first setup, compatibility checks, or lower-risk debugging.

Common issues:
- If the dropdown is empty, the model folder is missing or incomplete.
- If loading fails, verify that config.json and model weights exist in models/VoxCPM2.

### 2. VoxCPM Multilingual TTS

Purpose:
Direct text to speech in any officially supported language.

Inputs:
- model
- text
- cfg_value
- inference_timesteps
- max_len
- normalize
- denoise

How to use:
1. Connect the model output from Load VoxCPM Model.
2. Enter the text you want spoken.
3. Start with the default generation settings.
4. Send the output audio to your audio preview or save node.

Recommended usage:
- Use this node for English, Chinese, Japanese, French, Arabic, and other standard multilingual TTS tasks.
- Use it for dialects too, but write the actual dialect text when possible.

Example inputs:
- English: Hello, welcome to VoxCPM2.
- Chinese: 你好，欢迎使用 VoxCPM2。
- Cantonese: 伙计，唔该来一个 A 餐，冻奶茶少甜。

Parameter guidance:
- cfg_value: higher usually follows the prompt style more closely
- inference_timesteps: higher usually costs more time
- max_len: increase if the generated audio is cut too early
- normalize: useful for numbers, dates, abbreviations
- denoise: usually not needed here because there is no reference audio

Common issues:
- If pronunciation is not ideal for a dialect, rewrite the input into that dialect instead of standard Mandarin.
- If the output is cut off, increase max_len.

### 3. VoxCPM Voice Design

Purpose:
Create a brand-new synthetic voice from a natural-language description.

Inputs:
- model
- text
- voice_description
- cfg_value
- inference_timesteps
- max_len
- normalize
- denoise

How to use:
1. Connect the model output.
2. Put the target speech content into text.
3. Describe the voice you want in voice_description.
4. Run the node.

What voice_description should contain:
- gender
- age impression
- tone
- pace
- emotion
- style cues
- dialect cue if needed

Good examples:
- young woman, gentle and sweet voice, medium speaking pace
- middle aged male, calm documentary narration, low voice
- 粤语，中年男性，语气平淡
- 河南话，接地气的大叔，语速稍快

Bad examples:
- good voice
- normal person
- random style

Common issues:
- If the result is too generic, make the description more specific.
- If you want to clone a real speaker, use Controllable Cloning instead of Voice Design.

### 4. VoxCPM Controllable Cloning

Purpose:
Clone a speaker from a short reference clip and optionally steer style, emotion, or pace.

Inputs:
- model
- reference_audio
- text
- style_instruction
- cfg_value
- inference_timesteps
- max_len
- normalize
- denoise

How to use:
1. Connect the model output.
2. Connect a clean reference audio clip.
3. Enter the target text.
4. Optionally fill style_instruction to adjust expression.
5. Run the node.

What reference_audio should be like:
- single speaker
- clean speech
- low background noise
- no heavy music
- no overlapping voices

Good style_instruction examples:
- slightly faster, cheerful tone
- calm and steady narration
- emotional, softer ending, slower pace
- 粤语口吻，轻松一点，像朋友聊天

Recommended use cases:
- You want the speaker timbre from a real sample.
- You want more control than plain cloning.
- You do not have an exact transcript for the reference audio.

Common issues:
- If the reference is noisy, enable denoise and also load the model with load_denoiser enabled.
- If the result sounds off, try a cleaner and shorter reference clip.
- If you need the highest similarity and you also know the transcript, use Ultimate Cloning instead.

### 5. VoxCPM Ultimate Cloning

Purpose:
High-fidelity continuation cloning using prompt audio plus the exact transcript of that prompt audio.

Inputs:
- model
- prompt_audio
- prompt_text
- text
- optional reference_audio
- cfg_value
- inference_timesteps
- max_len
- normalize
- denoise

How to use:
1. Connect the model output.
2. Connect the prompt audio.
3. In prompt_text, write exactly what the prompt audio says.
4. In text, write what the speaker should continue saying next.
5. Optionally connect the same speaker again as reference_audio to reinforce similarity.
6. Run the node.

How to understand the inputs:
- prompt_audio: the spoken prefix
- prompt_text: transcript of the spoken prefix
- text: the new continuation to generate
- reference_audio: optional extra timbre reference

Best use case:
- You want the strongest possible cloning quality.
- You know the exact transcript of the prompt audio.
- You want the output to feel like the same speaker continues speaking naturally.

Very important:
- prompt_text must match prompt_audio closely
- wrong transcript quality directly hurts cloning quality

Example:
- prompt_audio says: 大家好，今天我们来聊一下配音技巧。
- prompt_text: 大家好，今天我们来聊一下配音技巧。
- text: 下面我会用三个例子说明停连和重音。

Common issues:
- If prompt_text does not match the prompt audio, the result can degrade a lot.
- If the prompt audio is too noisy, enable denoise and use a cleaner clip.
- If the output is too short, increase max_len.

### 6. VoxCPM Generate Audio Advanced

Purpose:
Expose the major VoxCPM generation inputs in a single flexible node.

Inputs:
- model
- text
- control_instruction
- optional reference_audio
- optional prompt_audio
- optional prompt_text
- cfg_value
- inference_timesteps
- max_len
- normalize
- denoise

Who should use it:
- Users who already understand the VoxCPM modes
- Users who want to build one node around multiple generation patterns
- Users who want a single advanced node instead of several specialized ones

How different input combinations map to modes:
- text only: plain multilingual TTS
- text plus control_instruction: voice design
- text plus reference_audio: cloning
- text plus reference_audio plus control_instruction: controllable cloning
- text plus prompt_audio plus prompt_text: continuation style cloning
- text plus prompt_audio plus prompt_text plus reference_audio: strongest advanced cloning setup

Recommended rule:
- If you are new, use the specialized nodes first.
- Use the Advanced node only after you clearly understand what each input does.

## Typical Workflows

### Workflow A: Basic multilingual TTS

Use:
- Load VoxCPM Model
- VoxCPM Multilingual TTS

### Workflow B: Create a brand-new voice

Use:
- Load VoxCPM Model
- VoxCPM Voice Design

### Workflow C: Clone a speaker and control style

Use:
- Load VoxCPM Model
- VoxCPM Controllable Cloning

### Workflow D: Highest-fidelity continuation cloning

Use:
- Load VoxCPM Model
- VoxCPM Ultimate Cloning

## Tips

- cfg_value controls how strongly the output follows the prompt or reference style
- inference_timesteps trades speed for quality
- normalize helps with numbers, dates, abbreviations, and text cleanup
- denoise is useful when the reference audio is noisy
- for Chinese dialects, writing the actual dialect text usually works better than plain Mandarin text

## Troubleshooting

### The node cannot find the model

Check that the model files are in:

    ComfyUI/models/VoxCPM2

and that config.json plus model weights are present.

### Import error for voxcpm

Either:
- install voxcpm into the current environment
- or place the official upstream repo at ComfyUI/VoxCPM

### Missing normalization or denoiser dependencies

If normalize or load_denoiser fails, you are probably missing optional dependencies such as:

- wetext
- inflect
- regex
- modelscope

## License

This custom node wrapper is built around the official VoxCPM project.

According to the official project page, VoxCPM model weights and code are released under the Apache-2.0 license.

Please review the upstream repository and model page for the exact current license terms.

## Acknowledgments

- https://github.com/OpenBMB/VoxCPM.git
- https://huggingface.co/openbmb/VoxCPM2
