# nlp_data_aug
Experiments with NLP Data Augmentation

## Data Augmentation Techniques
(some ideas from this [thread](https://forums.fast.ai/t/nlp-data-augmentation-experiments/39902) in fast.ai forums)
- Noun,Verb and Adjective replacement in the IMDB dataset from the [EDA paper](https://arxiv.org/abs/1901.11196)
- Pronoun replacement to come soon
- "Back Translation": Translating from one language to another and then back to the original to utilize the “noise” in back translation as augmented text.
- ~~Character Perturbation for augmentation (more from Mike)~~ Surface Pattern Perturbation
  Because ULMFit doesn't model characters, I will switch to surface patterns of word tokens instead. UNK token perturbation will be committed soon.

## Issues with the codebase
- ~~Cuda dropout non-deterministic?? (more from Mike)~~ Most of cuDNN nondeterministic issues are almost implicitly solved with fastai
  * For dropout, a deterministic usage is like what is done [fastai-1.0.55: awd_lstm.py#L105](https://github.com/fastai/fastai/blob/release-1.0.55/fastai/text/models/awd_lstm.py#L105). A non-determinstic way is to apply `torch.nn.LSTM(..., dropout=dropout_prob_here,...)` directly, although it is much faster.
  * Other cuDNN/PyTorch reproducibility issues are addressed in [#1-control_random_factors/imdb.ipynb](anz9990/nlp_data_aug/blob/%231-control_random_factors/imdb.ipynb). They require explicit measures, as https://docs.fast.ai/dev/test.html#getting-reproducible-results briefly described.

## Tasks
- Use data augmentation techniques in succession on IMDB classification task to report performance
- Make use of baseline translation model from [OpenNMT](http://opennmt.net/Models-py/) for "back translation" task

## Misc
- [Blog post](http://blog.aylien.com/research-directions-at-aylien-in-nlp-and-transfer-learning/#taskindependentdataaugmentationfornlp) on research directions for data augmentation in NLP
