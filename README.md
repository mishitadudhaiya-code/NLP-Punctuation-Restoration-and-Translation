# Hindi Punctuation Restoration, Sentence Ordering and Translation

Research assignment: add punctuation to jumbled Hindi sentences (Task 1), find the sentence order, group every 5 ordered sentences and translate each group to English (Task 2).

**Constraints followed:** no off-the-shelf LLM (GPT / Gemini / Claude etc.), and every model stays under **1.5 GB of GPU memory**.

## Links

- **Google Drive (outputs + notebooks):** [https://drive.google.com/drive/folders/1W2ptEekofK0he1zFgopUPiRpgwSfTtcL?usp=sharing)](https://drive.google.com/drive/folders/1W2ptEekofK0he1zFgopUPiRpgwSfTtcL?usp=sharing)

## Repository / Drive structure

```
Research_Project/
├── Model_1/punctuation_outputs/     # Task 1: sample-N_punctuated.txt (7 files)
├── Model_3/Sentence_Ordering_v2/    # Task 2 results
│   ├── sampleN_ordered.txt          # sentences in the predicted order
│   ├── sampleN_final.txt / .json    # chunks of 5: Hindi + English translation
│   ├── gpu_memory_report.csv        # measured GPU memory of every stage
│   ├── ordering_evaluation.csv      # ordering quality on held-out text
│   ├── review_flags.csv             # sentences / chunks that need a human look
│   └── run_info.json                # settings used in the run
└── NLP_Task2_FINAL_submit_v2.ipynb  # Task 2 notebook
```

## Task 1: Punctuation

I restored punctuation with the Cadence-Fast punctuation model (about 0.3B parameters, Gemma-3-270M base). It is a small model trained for this job, not a general-purpose LLM. I run it with half-precision weights so that it stays under the 1.5 GB limit. Outputs are in `Model_1/punctuation_outputs/`.

## Task 2: Sentence ordering and translation

Input: the punctuated sentences of each sample, shuffled, with `NULL` placeholders removed.

### Step 1: Understand the text (MiniLM + TF-IDF)
- I embed each sentence with `paraphrase-multilingual-MiniLM-L12-v2` (fp16).
- I also build character and word TF-IDF vectors (Hindi-aware tokenisation) to capture shared vocabulary.
- Text is Unicode-NFC normalised, because the files mix two encodings of letters like ढ़.

### Step 2: Cluster by topic
- I use Ward hierarchical clustering on the embeddings, with the number of clusters (1 to 3) chosen by silhouette score. A minimum silhouette (0.08) and a minimum cluster size prevent over-splitting.
- This separates samples that mix several articles.

### Step 3: Order the sentences inside each cluster
Two scorers are combined:
1. **Rules:** discourse cues (opening phrases, "इसलिए", "उसके बाद", etc.), pronoun and reference hints, and similarity between neighbours.
2. **Learned pair model:** a MiniLM cross-encoder fine-tuned on Hindi Wikipedia (streamed, no LLM) to answer "does sentence B come right after sentence A?". Its large word-embedding table is frozen and kept on the CPU to save GPU memory. A small logistic model additionally predicts which sentence starts a text.

The pair scores feed a greedy path search followed by insertion-based local search. The final method (rules only, learned only, or a mix) is picked on half of the evaluation windows and reported on the other half.

### Step 4: Chunk and translate
- I cut the ordered sentences into chunks of exactly 5.
- I translate each sentence separately and join the results per chunk.
- A hand-made Hindi-to-English glossary fixes place names and technical terms (e.g. Garhmukteshwar, Kanchipuram, tussar silk, cocoon).
- Translator selection is automatic. NLLB-600M (8-bit, fp16, 4-bit) is tried first, then M2M100-418M. A model is only accepted if its measured whole-GPU use is below 1.5 GB.

## GPU memory (measured on a Colab T4)

I measured memory as both the PyTorch peak and the whole GPU including the CUDA context. A hard cap of 1.5 GB was also set with `torch.cuda.set_per_process_memory_fraction`. One model is loaded at a time.

| Stage | PyTorch peak | Whole GPU |
|---|---|---|
| MiniLM embeddings | 0.62 GB | 0.76 GB |
| Ordering cross-encoder (train + score) | 0.80 GB | 1.39 GB |
| Translation (M2M100-418M fp16) | 1.13 GB | 1.40 GB |

All stages are under 1.5 GB by both measures (see `gpu_memory_report.csv`).

_Note: I used an AI assistant to help with coding and debugging while building the notebook. The pipeline itself uses no LLM, only the small models described above._

## Results and honest limitations

**Ordering** (shuffled 10-sentence windows from held-out Wikipedia text, Kendall tau, higher is better):
- Random: about 0.02
- Rules only: 0.071
- Final method (learned model + rules): 0.087
- The learned pair model reached AUC 0.67 and 52.8% direction accuracy (50% is guessing). It learned only weakly in the training budget, so the gain over the rules is small.

**Translation:**
- NLLB-600M gives better English, but it did not fit: 8-bit and 4-bit loading ran out of memory under the cap, and fp16 measured 1.60 GB for the whole GPU. It was rejected to respect the limit.
- M2M100-418M (1.40 GB) was used instead. It still makes errors: some rivers and names are mistranslated, and a few words are left in Devanagari. The glossary fixes only the names it covers. `review_flags.csv` lists the weakest cases.

## How I got here (development process)

1. **First version:** I started with MiniLM embeddings + TF-IDF + clustering + rule-based ordering, translated with `opus-mt-hi-en`. Ordering was reasonable, but the translation was bad (for example it translated the river Ganga as "Milky Way") and clustering split articles into too many pieces.
2. **Translation fixes:** I switched to NLLB-200-distilled-600M with M2M100-418M as a fallback, and added a Hindi-to-English glossary and sentence-by-sentence translation. IndicTrans2 was dropped because it failed to import with the Colab transformers version.
3. **Clustering fixes:** I added a minimum silhouette score, a maximum of 3 clusters and a minimum cluster size, so one article is no longer split into pieces.
4. **Bug fixes:** Unicode NFC normalisation (the files mix two encodings of ढ़, which broke glossary matching), and regex escapes in the English clean-up step.
5. **Learned ordering, attempt 1:** logistic regression on MiniLM embeddings. It failed (51.8% direction accuracy) because the embeddings are symmetric and cannot tell which sentence comes first.
6. **Learned ordering, attempt 2 (final):** a cross-encoder that reads both sentences together and is fine-tuned on Hindi Wikipedia. The word-embedding table is frozen and kept on the CPU to save GPU memory. It performs better, but it is still weak.
7. **Memory compliance:** I added a hard 1.5 GB cap, per-stage memory reports that include the CUDA context, one model loaded at a time, and automatic translator selection that rejects any model measured over 1.5 GB. This is why the NLLB-600M attempts were rejected.

## Known problems and fixes that can be made

| Problem | Why it happens | Possible fix |
|---|---|---|
| Translation quality (wrong names, leftover Devanagari, odd phrases) | NLLB-600M did not fit under 1.5 GB (fp16 measured 1.60 GB whole GPU), so the smaller M2M100-418M was used | Trim NLLB's 256k vocabulary to Hindi and English tokens only, use a smaller batch and fewer beams, then re-measure. Or load NLLB's embedding table on the CPU as I did for MiniLM |
| Names and technical terms mistranslated | The glossary only covers terms seen in the 7 samples | Extend the glossary, or transliterate unknown proper nouns instead of translating them |
| Leftover words like "Sriकृष्ण" | The model copies source words it does not know | Post-process: replace remaining Devanagari words with transliteration, or retry that sentence with a different beam setting |
| Ordering gain over rules is small (tau 0.087 vs 0.071) | The pair model learned weakly: loss stayed near 0.97 and direction accuracy was 52.8% | Train longer on more Wikipedia text, use a higher learning rate, train with a listwise or position-prediction loss, and add hard negatives from the same article |
| Next-sentence prediction is hard in general | Many Wikipedia sentences can follow each other plausibly | Combine with entity tracking (a person is named before "he/she"), date and number order, and discourse markers |
| Mixed-topic samples | Clustering depends on a fixed silhouette threshold | Tune the threshold on labelled mixes of articles, or use sentence-to-title similarity |
| Task 1 memory margin | Cadence-Fast in fp32 is about 1.1-1.2 GB, close to the limit | Load in bf16/fp16 and report `torch.cuda.max_memory_allocated()` |
| Not verified by a human | No gold translations or gold order were available | Check `review_flags.csv`, and have a Hindi speaker rate a sample of chunks |

## How to run

1. Put the 7 punctuated input files in `My Drive/Research_Project/Model_1/punctuation_outputs/`.
2. Open `NLP_Task2_FINAL_submit_v2.ipynb` in Google Colab with a GPU runtime (T4).
3. Run all cells. Outputs are written to `Research_Project/Model_3/Sentence_Ordering_v2/`.

Main libraries: PyTorch, transformers, sentence-transformers, scikit-learn, scipy, datasets, bitsandbytes.
