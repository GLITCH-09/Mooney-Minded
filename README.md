# Mooney-Minded

*Computational Visual Perception*, FAU, Project Part 1: can pretrained vision systems see what humans see in Mooney images?

100 COCO photos (10 classes × 10) become Mooney images with one global setting (Gaussian blur σ = 2.5 px, 55 % white pixels). Then they go through:

| | System | Question |
|---|---|---|
| Result1 | **Qwen3-VL-4B-Instruct**, local, all 100 images<br>**Qwen4-Max**, chat.qwen.ai, top 10, same prompts | Can it describe and name the object? Can it be talked into a wrong one? |
| Result2 | **SAM 3**, local, all 100 images | Given the class name, does it find and outline the object? Given a box, can it outline it? Does it hallucinate a wrong class? |

## Results

| | Photos | Mooneys |
|---|---|---|
| Qwen3-VL-4B names the object (all 100) | 97 % | 1 % |
| Qwen3-VL-4B picks it from 10 classes (all 100) | 100 % | 11 % |
| Qwen3-VL-4B / Qwen4-Max name it (top 10) | 10/10 / not run | 0/10 / 2/10 |
| "Yes" to a wrong class: Qwen3-VL-4B (all 100) / Qwen4-Max (top 10) | 0 % / not run | 6 % / 0/10 |
| SAM 3 finds it from the class name (all 100) | 100 % | 3 % |
| SAM 3 outlines it from a box, IoU ≥ 0.5 (all 100) | 100 % | 87 % |

Qwen4-Max was asked each question in a fresh chat with blind file names; raw replies are in `results/qwen4max_web.txt`.

Everything runs from the repository root as modules. Images, outputs and weights stay local (`data/`, `outputs/`).

## Setup

```bash
python3 -m venv .venv-tool && source .venv-tool/bin/activate && pip install -r requirements.txt

python3 -m venv .venv-vlm && source .venv-vlm/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.57" accelerate pillow opencv-python

uv venv .venv-sam3 --python 3.12 && source .venv-sam3/bin/activate
uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3 ../sam3
uv pip install -e ../sam3 einops psutil "setuptools<80" pycocotools scikit-image scipy "opencv-python<4.12"
hf auth login
```

## Run

```bash
# .venv-tool: dataset and Mooneys
python -m scripts.prepare_dataset
python -m mooney_tool.gui                      # pick one setting by eye
python -m mooney_tool.cli -i data/Original -o data/Mooney --blur 2.5 --white 55

# .venv-vlm: Qwen3-VL-4B
python -m vision.qwen --split Original
python -m vision.qwen --split Mooney

# .venv-sam3: SAM 3
python -m vision.sam --split Original
python -m vision.sam --split Mooney

# .venv-tool: Qwen4-Max by hand, then figures and the submission zip
python -m vision.web blind && python -m vision.web sheet   # paste replies into results/qwen4max_web.txt
python -m vision.web score
python -m scripts.figures
python -m scripts.package --pptx MooneyMinded_Part1.pptx
```

## Layout

```
mooney_tool/   Mooney generator: core, CLI, Gradio GUI
vision/        common (labels, prompts, scoring), qwen (local VLM), web (Qwen4-Max sheet), sam (SAM 3)
scripts/       prepare_dataset, figures, package
results/       scored tables per model and split (committed)
```
