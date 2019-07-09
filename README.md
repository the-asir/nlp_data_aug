# nlp_data_aug
Experiments with NLP Data Augmentation

## Data Augmentation Techniques
(some ideas from this [thread](https://forums.fast.ai/t/nlp-data-augmentation-experiments/39902) in fast.ai forums)
- Noun,Verb and Adjective replacement in the IMDB dataset from the [EDA paper](https://arxiv.org/abs/1901.11196)
- Pronoun replacement to come soon
- "Back Translation": Translating from one language to another and then back to the original to utilize the “noise” in back translation as augmented text.
- Character Perturbation for augmentation (more from Mike)

## Issues with the codebase
- Cuda dropout non-deterministic?? (more from Mike)

## Tasks
- Use data augmentation techniques in succession on IMDB classification task to report performance
- Make use of baseline translation model for "back translation" task

## Misc
- [Blog post](http://blog.aylien.com/research-directions-at-aylien-in-nlp-and-transfer-learning/#taskindependentdataaugmentationfornlp) on research directions for data augmentation in NLP
