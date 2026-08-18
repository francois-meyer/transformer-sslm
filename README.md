# Transformer SSLM

Code for pretraining and finetuning the Transformer-based SSLM (subword segmental language model) proposed in the paper [*The Learning Dynamics of Subword Segmentation for Morphologically Diverse Languages*](https://aclanthology.org/2025.ijcnlp-long.36/). During pretraining, the model learns a subword segmentation scheme that optimises its autoregressive language modelling objective. During finetuning, the model adapts its subword segmentation to optimise text generation in a downstream task.

![](sslm-arch.png)

The model is implemented as an architecture in fairseq. This repo contains code for pretraining, completion-only finetuning, and text generation. We will publicly release our pretrained SSLMs for Setswana, English, and isiXhosa upon publication.


## Dependencies
* python 3
* [fairseq](https://github.com/pytorch/fairseq) (commit: 806855bf660ea748ed7ffb42fe8dcc881ca3aca0)
* pytorch 1.0.1.post2
* cuda 11.4
* nltk

## Usage
Merge the SSLM files with fairseq.

```shell
git clone https://github.com/pytorch/fairseq.git
git clone https://github.com/francois-meyer/transformer-sslm

# change to 806855bf660ea748ed7ffb42fe8dcc881ca3aca0 branch
cd fairseq
git checkout 806855bf660ea748ed7ffb42fe8dcc881ca3aca0 

# copy files from ssmt to fairseq
cp -r ../transformer-sslm/fairseq ./ 
cp -r ../transformer-sslm/fairseq_cli ./  
```

## Data Format

The downstream task data for finetuning should be formatted as follows: 
`{input context}={correct output}`. For example `What is the capital of Ghana?=Accra`.

The equals sign (`=`) separates the input prompt from the expected prompt completion. During finetuning, `=` is used as a marker to delineate completion-only optimisation. The likelihood of all text before and including `=` will not be maximised, only the subsequent text likelihood will be maximised as a generation. This is known as [completion-only finetuning](https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only).

**Note:** `=` is used as a special token to mark input-output boundaries, so all other instances of the `=` character must be removed from input and output text.


**Data-to-text example**

As an example of how to format downstream task data for completion-only finetuning, we present our approach to finetuning SSLM for isiXhosa data-to-text generation. Each example in the data-to-text dataset consists of a triple `(subject, relation, object)` mapped to a descriptive isiXhosa sentence e.g. `(South Africa, leaderName, Cyril Ramaphosa)` maps to `uCyril Ramaphosa yinkokheli yoMzantsi Afrika` (translation: `Cyril Ramaphosa is the leader of South Africa`). To format this as a completed prompt, we flatten the triple into the text format `subject + relation + object`. We then concatenate ` # # =`, our template for separating input and output, which is followed by the expected text generation.   

The train and validation files should contain completed prompts `{input context}={correct output}`, while the test file should contain incomplete prompts up to the equals sign `{input context}=`. Train, validation, and test files should contain one prompt example per line.

Examples in the train and validation files:

```
South Africa + leaderName + Cyril Ramaphosa # # =uCyril Ramaphosa yinkokheli yomzantsi Afrika.
Ethiopia + currency + Ethiopian birr # # =Imali yase-Ethiopia yi-Ethiopian Birr.
Denmark + capital + Copenhagen # # =ICopenhagen likomkhulu laseDenmark.
``` 

Examples in the test file:

```
Romania + capital + Bucharest # # =
India + currency + Indian rupee # # =
Netherlands + leaderName + Mark Rutte # # =
```




## Instructions

1. Preprocess pretraining data.

```shell
python fairseq/fairseq_cli/preprocess.py  \
    --only-source \
    --trainpref $PT_DATA_DIR/train.xh --validpref $PT_DATA_DIR/valid.xh --testpref $PT_DATA_DIR/test.xh \
    --destdir $PT_DATA_DIR/bin
```

2. Pretrain T-SSLM.

```shell
python fairseq/fairseq_cli/train.py $PT_DATA_DIR \
    --task subword_segmental_language_modeling \
    --arch transformer_sslm \
    --target-lang xh --save-interval 5 \
    --criterion subword_segmental_lm_cross_entropy \
    --max-epoch 40 --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.1 --skip-invalid-size-inputs-valid-test \
    --tokens-per-sample 512 --sample-break-mode none \
    --max-tokens 4096 --vocabs-path $PT_MODEL_DIR --update-freq 64 \
    --keep-best-checkpoints 1  \
    --max-seg-len 5 --lexicon-max-size 10000 \
    --save-dir $PT_MODEL_DIR &>> $PT_MODEL_DIR/log


```

3. Preprocess finetuning data.

```shell
python fairseq/fairseq_cli/preprocess.py  \
    --only-source --srcdict $PT_DATA_DIR/bin/dict.txt \
    --trainpref $FT_DATA_DIR/train.xh --validpref $FT_DATA_DIR/valid.xh --testpref $FT_DATA_DIR/test.xh \
    --destdir $FT_DATA_DIR/bin
```

4. Finetune T-SSLM for completion-only text generation.



```shell
python fairseq/fairseq_cli/train.py $FT_DATA_DIR \
    --task subword_segmental_language_modeling \
    --arch transformer_sslm \
    --target-lang xh \
    --max-epoch 20 --share-decoder-input-output-embed \
    --save-interval 10 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 500 --label-smoothing 0.1 \
    --dropout 0.3 --skip-invalid-size-inputs-valid-test \
    --batch-size 16 --tokens-per-sample 512 --sample-break-mode none \
    --vocabs-path $PT_MODEL_DIR --update-freq 1 \
    --no-epoch-checkpoints --keep-best-checkpoints 1  \
    --max-seg-len 5 --lexicon-max-size 10000 \
    --criterion prompt_subword_segmental_lm_cross_entropy --line-prompts \
    --finetune-from-model $PT_MODEL_DIR/checkpoint_best.pt \
    --save-dir $FT_MODEL_DIR &>> $FT_MODEL_DIR/log

```

5. Generate text with a finetuned T-SSLM. Set `--decoding dynamic` to use the recommended dynamic decoding (character-by-character generation) or set `--decoding separate` to use unmixed decoding (subword-level generation).

```shell
python fairseq/fairseq_cli/generate_sslm.py $DATA_DIR \
    --task subword_segmental_language_modeling \
    --path $FT_MODEL_DIR/checkpoint_best.pt \
    --decoding dynamic --marginalize none-none \
    --arch transformer_sslm \
    --target-lang xh \
    --skip-invalid-size-inputs-valid-test \
    --tokens-per-sample 512 --sample-break-mode none \
    --max-tokens 4096 --vocabs-path $PT_DIR \
    --max-seg-len 5 --lexicon-max-size 10000 \
    --line-prompts --batch-size 1 --beam 5 \
    --normalize-type seg-seg  \
    --results-path $RESULTS_DIR &>> $RESULTS_DIR/log
```

### Citation

```bibtex
@inproceedings{meyer-buys-2025-learning,
    title = "The Learning Dynamics of Subword Segmentation for Morphologically Diverse Languages",
    author = "Meyer, Francois  and
      Buys, Jan",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.36/",
    doi = "10.18653/v1/2025.ijcnlp-long.36",
    pages = "647--661",
    ISBN = "979-8-89176-298-5",
    abstract = "Subword segmentation is typically applied in preprocessing and stays fixed during training. Alternatively, it can be learned during training to optimise the training objective. In this paper we study the learning dynamics of subword segmentation: if a language model can dynamically optimise tokenisation, how do its subwords evolve during pretraining and finetuning? To explore this, we extend the subword segmental language model (SSLM), a framework for learning subwords during training, to support pretraining and finetuning. We train models for three typologically diverse languages to study learning dynamics across the morphological spectrum: Isi-Xhosa is conjunctive (long word forms composed of many morphemes), Setswana is disjunctive (morphemes written as separate words), and English represents a typological middle ground. We analyse subword dynamics from a linguistic perspective, tracking morphology, productivity, and fertility. We identify four stages of subword learning, with the morphologically complex isi-Xhosa exhibiting greater instability. During finetuning, subword boundaries shift to become finer-grained. Lastly, we show that learnable subwords offers a promising approach to improve text generation and cross-lingual transfer for low-resource, morphologically complex languages."
}
```
