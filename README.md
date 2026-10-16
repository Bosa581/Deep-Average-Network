# Quizbowl Question Classification with a Deep Averaging Network

A PyTorch text classifier that predicts the category of a quizbowl question, such as History, Literature, or Science. The model learns word embeddings, averages them into a question representation, and passes that representation through a small feedforward network to predict one of 11 categories.

## How it works

```text
Question text → NLTK tokenization → Word embeddings → Mean pooling
              → Linear layer → ReLU → Dropout → Category scores
```

- **Embeddings:** learned from scratch, with 50 dimensions per token.
- **Classifier:** a 50-unit hidden layer and dropout of 0.5.
- **Variable-length inputs:** padded batches with averages computed using actual question lengths. Padding embeddings are zero; unknown words map to `<unk>`.
- **Training:** cross-entropy loss, Adamax optimization, and gradient clipping at 5.
- **Checkpoint selection:** saves the model with the best development accuracy and uses that checkpoint for final evaluation.
- **Hardware:** automatically uses CUDA when available, with a CPU option.

## Setup

Install Python and the dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and NLTK tokenizer resources:

```bash
python -m pip install torch numpy nltk
python -m nltk.downloader punkt punkt_tab
```

## Dataset

The classifier uses QANTA quizbowl question data, split into training, development, and test sets. Place the files at:

```text
data/
├── qanta.train.json
├── qanta.dev.json
└── qanta.test.json
```

The training file is excluded from Git and must be supplied separately. Development and test files are included in the repository. Each file must contain a top-level `questions` array whose entries include `text` and `category`:

```json
{
  "questions": [
    {"text": "Which planet is known as the Red Planet?", "category": "Science"}
  ]
}
```

Entries without a category are skipped. All three dataset files are loaded at startup, including when evaluating a saved model. Use `--train-file`, `--dev-file`, and `--test-file` to provide alternative paths.

## Train

Run with the default configuration of 20 epochs and a batch size of 128:

```bash
python dan.py
```

The best development checkpoint is saved to `q_type.pt`. After training, the script reloads it and prints training, development, and test accuracy.

Customize a run or force CPU execution:

```bash
python dan.py --num-epochs 20 --batch-size 128 --no-cuda --save-model q_type.pt
```

Continue training from a saved model:

```bash
python dan.py --resume --load-model q_type.pt --save-model q_type.pt
```

Resuming restores the model and its vocabulary/category mappings; the optimizer is initialized again.

## Evaluate and inspect errors

Evaluate test accuracy and display up to 10 incorrect predictions:

```bash
python dan.py --test --load-model q_type.pt --max-errors 10
```

Inspect development errors or evaluate training accuracy:

```bash
python dan.py --dev-errors --load-model q_type.pt --max-errors 10
python dan.py --train-eval --load-model q_type.pt
```

Each displayed error includes the question, true category, and predicted category. For all command-line options:

```bash
python dan.py --help
```

## Reported results

The recorded experiment used 20 epochs, a batch size of 128, and learned embeddings without pretrained GloVe vectors. Accuracy for its best development checkpoint was:

| Split | Accuracy |
| --- | ---: |
| Training | 88.62% |
| Development | 90.30% |
| Test | 89.67% |

These are recorded results, rather than a guarantee for a new run. Training does not set a fixed random seed.

Observed development errors included a film question labeled Trash but predicted as Fine Arts, and a question about burial rituals labeled Religion but predicted as Mythology. Shared vocabulary between categories may contribute to these mistakes.

## Tests

```bash
python -m unittest test.py
```

The tests cover token vectorization, unknown-word handling, forward-pass scores, probability outputs, and padded minibatches.

## Limitations

- Mean pooling discards word order and gives every token equal weight.
- Unknown words share one embedding, which can obscure unfamiliar names and clues.
- Long questions can dilute informative words with less useful text.
- Similar vocabulary across categories can lead to ambiguous predictions.

## Project files

| File | Purpose |
| --- | --- |
| `dan.py` | Data loading, model architecture, training, evaluation, and error inspection |
| `test.py` | Unit tests for vectorization and model behavior |
| `data/` | Development and test datasets; location for the separately supplied training data |
| `q_type.pt` | Saved model checkpoint |
