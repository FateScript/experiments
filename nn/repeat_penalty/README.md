## Repeat penalty

This is a simple implementation of a repeat penalty for the Transformer model.

### Files

`gpt.py`: The main file that contains the implementation of a baby gpt.  
`tokenizer.py`: super simple implementation of the tokenizer.  
`repeat_penalty.py` shows why LLM has repeat outputs and how repeat penalty can help.


### Usage

To see the repeat outputs of the model, run the following command:
```bash
python3 repeat_penalty.py
```

To see the effect of the repeat penalty, run the following command:
```bash
python3 repeat_penalty.py --repeat_penalty 1.1
```

For English text, add `--en` flag in your command.
