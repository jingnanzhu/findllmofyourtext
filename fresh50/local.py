"""Generate matched Qwen answers, then judge anonymized Gemini/Qwen pairs."""
import json
import random
from pathlib import Path

from mlx_vlm import generate
from mlx_vlm.utils import load_model, load_tokenizer
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
gemini = json.loads((root / 'gemini.json').read_text())
snapshots = Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen2.5-VL-7B-Instruct-4bit/snapshots'
snapshot = next(snapshots.iterdir())
model = load_model(snapshot)
processor = AutoTokenizer.from_pretrained(snapshot)
processor.detokenizer = load_tokenizer(snapshot, return_tokenizer=False)(processor)
def ask(prompt, limit):
    template = processor.apply_chat_template([dict(role='user',content=prompt)], tokenize=False, add_generation_prompt=True)
    result = generate(model, processor, template, max_tokens=limit, temperature=0, verbose=False)
    return getattr(result, 'text', result).strip()

def save(name, items):
    (root / name).write_text(json.dumps(items, ensure_ascii=False, indent=2))

path = root / 'qwen.json'
qwen = json.loads(path.read_text()) if path.exists() else []
for item in gemini[len(qwen):]:
    answer = ask('Give one independent answer in English of 35–55 words, one paragraph. No heading, introduction, or sign-off.\n'+item['prompt'], 140)
    qwen.append(dict(prompt=item['prompt'], text=answer))
    save('qwen.json', qwen)

rng = random.Random(80013)
order = list(range(25))
rng.shuffle(order)
pairs, key = [], []
for index in order:
    options = [gemini[index]['text'], qwen[index]['text']]
    reverse = rng.randrange(2)
    if reverse:
        options.reverse()
    pairs.append(dict(prompt=gemini[index]['prompt'], A=options[0], B=options[1]))
    key.append('B' if reverse else 'A')
save('pairs.json', pairs)
save('key.json', key)
predictions = []
for pair in pairs:
    reply = ask('One answer is from Gemini and the other from Qwen2.5-VL-7B. Which is Gemini? Answer with A or B only.\n'+json.dumps(pair,ensure_ascii=False), 12)
    predictions.append(reply)
save('qwen-guesses.json', predictions)
reason = ask("Which observable writing habits did you use to tell Gemini from Qwen? Inspect these anonymous examples and your predictions. Give two testable hypotheses, acknowledging uncertainty.\n" + json.dumps([dict(pair=p, predicted_gemini=g) for p,g in zip(pairs[:3], predictions[:3])],ensure_ascii=False), 190)
save('qwen-reason.json',reason)
