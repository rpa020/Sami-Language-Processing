## Empowering Sami Language Processing: A Foundational Model Approach for Low-Resource Languages

Sami, an indigenous language group comprising multiple languages, faces
digital marginalization due to the limited data availability and sophisticated
language models designed for its linguistic intricacies. This work focuses on
increasing technological participation in the Sami languages. We draw the
attention of the machine learning community towards the language modeling
problem of Ultra Low Resource (ULR) languages. ULRL refer to languages that
have a limited amount of available textual resources and a relatively low number
of speakers. These languages often lack extensive corpora, comprehensive data
across diverse domains, and sufficient linguistic research. Additionally, ULRLs
are unsupported by mainstream Large Language Models (LLMs) like ChatGPT,
so gathering artificial training data becomes problematic. With few speakers
of these languages, manual data creation becomes even more challenging.
However, it is essential to develop foundational models for these ULR languages
to promote inclusion and impact LLMs’ tangible abilities. To this end, we have
compiled the available Sami language resources from the web to create a clean
dataset for training language models.


To study the behavior of modern LLM models with ULR languages (Sami), we
have experimented with different kinds of LLMs, mainly at the order of seven
billion parameters. We have also explored the effect of multilingual LLM train-
ing for ULRLs, and trained models with different settings. With the produced
model, we explain how we can use it to implement a plagiarism detection
application. This project is the first study on the Sami language for adapting
non-statistical language models that use the latest natural language processing
(NLP) developments. We believe that the proposed dataset and findings from
this study are going to accelerate future research for ULRLs.


The experiments were done on three different decoder-only models and a
single sequence-to-sequence model. Table 3.2 shows the individual models
performance in cross-entropy loss, perplexity, self-BLEU and inference time.
Model S represents the sequence-to-sequence pegasus model with random
initialized weights, i.e., it has no prior knowledge. D1 represents the decoder-
only BLOOM model, also initialized with random weights. D2 and D3 are also
BLOOM models but have prior pretraining knowledge in Finnish. D2 is trained
with Northern Sami only, while D3 has performed joint multilingual training
in Northern Sami, Finnish, and Norwegian.

## Experiment Results
![Results Table](results_table.svg)
