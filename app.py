
import gradio as gr
import torch
import uuid
from mario_gpt.dataset import MarioDataset
from mario_gpt.prompter import Prompter
from mario_gpt.lm import MarioLM
from mario_gpt.utils import view_level, convert_level_to_png

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import os
import uvicorn

mario_lm = MarioLM()
device = torch.device('cuda')
mario_lm = mario_lm.to(device)
TILE_DIR = "data/tiles"

app = FastAPI()

def make_html_file(generated_level):
    level_text = f"""{'''
'''.join(view_level(generated_level,mario_lm.tokenizer))}"""
    unique_id = uuid.uuid1()
    with open(f"static/demo-{unique_id}.html", 'w', encoding='utf-8') as f:
        f.write(f'''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <title>Mario Game</title>
    <script src="https://cjrtnc.leaningtech.com/20230216/loader.js"></script>
</head>

<body>
</body>
<script>
    cheerpjInit().then(function () {{
        cheerpjAddStringFile("/str/mylevel.txt", `{level_text}`);
    }});
    cheerpjCreateDisplay(512, 500);
    cheerpjRunJar("/app/static/mario.jar");
</script>
</html>''')
    return f"demo-{unique_id}.html"

def generate(pipes, enemies, blocks, elevation, temperature = 2.0, level_size = 1399, prompt = ""):
    if prompt == "":
        prompt = f"{pipes} pipes, {enemies} enemies, {blocks} blocks, {elevation} elevation"
    print(f"Using prompt: {prompt}")
    prompts = [prompt]
    generated_level = mario_lm.sample(
        prompts=prompts,
        num_steps=level_size,
        temperature=temperature,
        use_tqdm=True
    )
    filename = make_html_file(generated_level)
    img = convert_level_to_png(generated_level.squeeze(), TILE_DIR, mario_lm.tokenizer)[0]
    
    gradio_html = f'''<div>
        <iframe width=512 height=512 style="margin: 0 auto" src="static/{filename}"></iframe>
        <p style="text-align:center">Press the arrow keys to move. Press <code>a</code> to run, <code>s</code> to jump and <code>d</code> to shoot fireflowers</p>
    </div>'''
    return [img, gradio_html]

with gr.Blocks().queue() as demo:
    gr.Markdown('''### Playable demo for MarioGPT: Open-Ended Text2Level Generation through Large Language Models
    [[Github](https://github.com/shyamsn97/mario-gpt)], [[Paper](https://arxiv.org/abs/2302.05981)]
    ''')
    with gr.Tabs():
        with gr.TabItem("Compose prompt"):
            with gr.Row():
                pipes = gr.Radio(["no", "little", "some", "many"], label="How many pipes?")
                enemies = gr.Radio(["no", "little", "some", "many"], label="How many enemies?")
            with gr.Row():
                blocks = gr.Radio(["little", "some", "many"], label="How many blocks?")
                elevation = gr.Radio(["low", "high"], label="Elevation?")
        with gr.TabItem("Type prompt"):
            text_prompt = gr.Textbox(value="", label="Enter your MarioGPT prompt. ex: 'many pipes, many enemies, some blocks, low elevation'")
        
    with gr.Accordion(label="Advanced settings", open=False):
        temperature = gr.Number(value=2.0, label="temperature: Increase these for more diverse, but lower quality, generations")
        level_size = gr.Slider(value=1399, precision=0, minimum=100, maximum=2799, step=1, label="level_size")
    
    btn = gr.Button("Generate level")
    with gr.Row():
        with gr.Box():
            level_play = gr.HTML()    
        level_image = gr.Image()
    btn.click(fn=generate, inputs=[pipes, enemies, blocks, elevation, temperature, level_size, text_prompt], outputs=[level_image, level_play])
    gr.Examples(
        examples=[
            ["many", "many", "some", "high"],
            ["no", "some", "many", "high", 2.0],
            ["many", "many", "little", "low", 2.0],
            ["no", "no", "many", "high", 2.4],
        ],
        inputs=[pipes, enemies, blocks, elevation],
        outputs=[level_image, level_play],
        fn=generate,
        cache_examples=True,
    )

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app = gr.mount_gradio_app(app, demo, "/", gradio_api_url="http://localhost:7860/")
uvicorn.run(app, host="0.0.0.0", port=7860)