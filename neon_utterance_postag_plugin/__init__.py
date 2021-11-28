# # NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# # All trademark and other rights reserved by their respective owners
# # Copyright 2008-2021 Neongecko.com Inc.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from neon_transformers import UtteranceTransformer
from neon_modelhub.postag.nltk_postag import get_default_postagger
from neon_modelhub import load_model
from os.path import isfile, expanduser
from quebra_frases import word_tokenize
import joblib


class PosTagger(UtteranceTransformer):
    """
        {
            "en": {
                "universal": "model_path/model_id",  # universal dependencies is the default tagset
                "treebank": "model_path/model_id",  # extra tagsets can be added in .conf
                "brown": "model_path/model_id"  # the name can be anything, used as key of returned context
            },
            "pt-br": {}, # full lang codes are matched first
            "pt": {}  # if full lang code not found
        }
    """
    def __init__(self, name="PosTagger", priority=5):
        super().__init__(name, priority)
        self.taggers = {}
        self.models = self.config.get("models") or {}
        self.load_models()

    def load_models(self, models=None):
        models = models or self.models
        for lang in models:
            self.models[lang] = self.models.get(lang) or {}
            for model_id, model_path in models[lang].items():
                self.models[lang][model_id] = model_path
                if isfile(expanduser(model_path)):
                    self.taggers[model_id] = joblib.load(expanduser(model_path))
                else:
                    self.taggers[model_id] = load_model(model_path)

    def get_taggers(self, lang):
        if lang in self.models:
            return {m: self.taggers[m] for m in self.models[lang]}
        lang = lang[:2]
        if lang in self.models:
            return {m: self.taggers[m] for m in self.models[lang]}
        # use whatever nltk model is recommended by neon_modelhub
        # nltk is used to keep minimal dependencies
        return {"universal": get_default_postagger(lang)}

    def transform(self, utterances, context=None):
        context = context or {}
        postagged = {}
        lang = context.get("lang") or self.config.get("lang", "en-us")
        for key, tagger in self.get_taggers(lang).items():
            postagged[key] = [tagger.tag(word_tokenize(utt))
                              for utt in utterances]
        return utterances, {"postag": postagged}
