# nlp_data_aug
Experiments with NLP Data Augmentation

## Data Augmentation Techniques
(some ideas from this [thread](https://forums.fast.ai/t/nlp-data-augmentation-experiments/39902) in fast.ai forums)
- Noun,Verb and Adjective replacement in the IMDB dataset from the [EDA paper](https://arxiv.org/abs/1901.11196)
- Pronoun replacement to come soon
- "Back Translation": Translating from one language to another and then back to the original to utilize the “noise” in back translation as augmented text.
- ~~Character Perturbation for augmentation (more from Mike)~~ Surface Pattern Perturbation
  * Because ULMFit doesn't model characters, I will switch to surface patterns of word tokens instead. UNK token perturbation will be committed soon.

## Issues with the codebase
- ~~Cuda dropout non-deterministic?? (more from Mike)~~ Most of cuDNN nondeterministic issues are almost implicitly solved with fastai
  * For dropout, a deterministic usage is like what is done at [fastai-1.0.55: awd_lstm.py#L105](https://github.com/fastai/fastai/blob/release-1.0.55/fastai/text/models/awd_lstm.py#L105). A non-determinstic way is to apply `torch.nn.LSTM(..., dropout=dropout_prob_here,...)` directly, although it is much faster.
  * Other cuDNN/PyTorch reproducibility issues are addressed in [#1-control_random_factors/imdb.ipynb](https://github.com/anz9990/nlp_data_aug/blob/%231-control_random_factors/imdb.ipynb). They require explicit measures, as https://docs.fast.ai/dev/test.html#getting-reproducible-results briefly described.
- Hyperparameters
  * from the ULMFiT paper (using pre-1.0 fastai):
    > We use the AWD-LSTM language model (Merity et al., 2017a) with an embedding size of 400, 3 layers, 1150 hidden activations per layer, and a BPTT batch size of 70. We apply dropout of 0.4 to layers, 0.3 to RNN layers, 0.4 to input embedding layers, 0.05 to embedding layers, and weight dropout of 0.5 to the RNN hidden-to-hidden matrix. The classifier has a hidden layer of size 50. We use Adam with β1 = 0.7 instead of the default β1 = 0.9 and β2 = 0.99, similar to (Dozat and Manning, 2017). We use a batch size of 64, a base learning rate of 0.004 and 0.01 for fine-tuning the LM and the classifier respectively, and tune the number of epochs on the validation set of each task.

    > On small datasets such as TREC-6, we fine-tune the LM only for 15 epochs without overfitting, while we can fine-tune longer on larger datasets. We found 50 epochs to be a good default for fine-tuning the classifier.

  * from [fastai/course-v3/nbs/dl1/lesson3-imdb.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb) (May 3, 2019)
    > learner|bs |lr  |bptt|wd  |clip|drop_mult|`to_fp16()`
    > --     |--:|--: |--: |--: |--: |--:      |--
    > lm     | 48|1e-2|  70|0.01|None|0.3      |No
    > cf     | 48|2e-2|  70|0.01|None|0.5      |No

  * from [fastai/fastai/examples/ULMFit.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/examples/ULMFit.ipynb) (June 11, 2019)
    > Fine-tuning a forward and backward langauge model to get to 95.4% accuracy on the IMDB movie reviews dataset. This tutorial is done with fastai v1.0.53.

    > The example was run on a Titan RTX (24 GB of RAM) so you will probably need to adjust the batch size accordinly. If you divide it by 2, don't forget to divide the learning rate by 2 as well in the following cells. You can also reduce a little bit the bptt to gain a bit of memory.

    > learner|bs |lr  |bptt|wd  |clip|drop_mult|`to_fp16()`
    > --     |--:|--: |--: |--: |--: |--:      |--
    > lm     |256|2e-2|  80|0.1 |0.1 |1.0      |Yes
    > cf     |128|1e-1|  80|0.1†|None|0.5      |No

    > † forward cf's wd used default value 0.01

  * from [fastai/course-nlp/nn-imdb-more.ipynb](https://nbviewer.jupyter.org/github/fastai/course-nlp/blob/master/nn-imdb-more.ipynb) (June 12, 2019)
    > learner|bs |lr        |bptt|wd  |clip|drop_mult|`to_fp16()`
    > --     |--:|--:       |--: |--: |--: |--:      |--
    > lm     |128|1e-2*bs/48|  70|0.01|None|1.0      |Yes
    > cf     |128|2e-2*bs/48|  70|0.01|None|0.5      |Yes

## Tasks
- Use data augmentation techniques in succession on IMDB classification task to report performance
- Make use of baseline translation model from [OpenNMT](http://opennmt.net/Models-py/) for "back translation" task

## Misc
- [Blog post](http://blog.aylien.com/research-directions-at-aylien-in-nlp-and-transfer-learning/#taskindependentdataaugmentationfornlp) on research directions for data augmentation in NLP
- For visual diff between Jupyter notebooks, try https://app.reviewnb.com
