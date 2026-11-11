# findllmofyourtext

Can language models recognize which model wrote a passage? This project contains 50 original English answers to 25 prompts, the anonymous answer pairs, and the saved judgments from Gemini Auto and Qwen2.5-VL-7B. Topics range from explaining how a loom works to describing a bakery during a power cut.

Run `python3 score.py` from this directory to verify the answer key against the original responses and reproduce the results. Python's standard library is sufficient.

| Judge | Correct pairs |
| --- | ---: |
| Gemini Auto | 25/25 |
| Qwen2.5-VL-7B | 9/25 |

Gemini answered the prompts in a single batch. Qwen answered them individually. The pair order and the position of each author were randomized. Each judge was told both candidate names and that each pair contained one answer from each. Gemini judged in a new chat. Its web interface identified the generation mode as Auto, so the exact underlying version is unknown.

The two-way test measures recognition within this sample. A rule that always chose the shorter response got 19/25 pairs right. The scores do not establish accuracy for arbitrary text or exact model versions.

The files in `fresh50/` contain the source responses, blind pairs, answer key, guesses and Gemini's explanation. The optional `local.py` script uses MLX on Apple Silicon with the cached Qwen model to generate missing local responses and rerun judging. Use `score.py` to inspect the original experiment without changing the files.
