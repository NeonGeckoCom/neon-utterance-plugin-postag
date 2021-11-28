```python
tagger = PosTagger()
_, context = tagger.transform(["o nome do meu cão é Jurebes VI, o temível"], {"lang": "pt"})
print(context)
# {'postag': {'universal': [[('o', 'DET'),
#                            ('nome', 'NOUN'),
#                            ('do', 'ADP'),
#                            ('meu', 'PRON'),
#                            ('cão', 'NOUN'),
#                            ('é', 'VERB'),
#                            ('Jurebes', 'NOUN'),
#                            ('VI', 'NOUN'),
#                            (',', '.'),
#                            ('o', 'DET'),
#                            ('temível', 'NOUN')]]}}

# you can enable new tagsets or override models in .conf
# model_id from https://github.com/NeonJarbas/modelhub
# or full path to local file
tagger.load_models({
    "en": {
        "treebank": "nltk_treebank_perceptron_tagger",  # extra tagsets can be added in .conf
        "brown": "nltk_brown_perceptron_tagger"  # the name can be anything, used as key of returned context
    }
})
_, context = tagger.transform(["My name is Casimiro"])
print(context)
# {'postag': {'brown': [[('My', 'PP$'),
#                        ('name', 'NN'),
#                        ('is', 'BEZ'),
#                        ('Casimiro', 'NP-HL')]],
#             'treebank': [[('My', 'NNP'),
#                           ('name', 'NN'),
#                           ('is', 'VBZ'),
#                           ('Casimiro', 'NNP')]]}}
```
