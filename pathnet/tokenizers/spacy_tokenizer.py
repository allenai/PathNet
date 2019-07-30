"""
Spacy package and english model is required.
"""

import spacy
import copy
from typing import List, Any, Set


class Tokens(object):
    """
    A class to represent a list of tokenized text.
    """
    TEXT = 0
    CHAR = 1
    TEXT_WS = 2
    SPAN = 3
    POS = 4
    LEMMA = 5
    NER = 6

    def __init__(self, data: List[Any], annotators: Set,
                 opts: Any = None, sents: Any = None) -> None:
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}
        self.sents = sents

    def __len__(self):
        """
        The number of tokens.
        """
        return len(self.data)

    def slice(self, i: int = None, j: int = None):
        """
        Return a view of the list of tokens from [i, j).
        """
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self) -> str:
        """
        Returns the original text (with whitespace reinserted).
        """
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased: bool = False) -> List[str]:
        """
        Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def sentences(self, uncased: bool = False) -> List[List[str]]:
        """
        Returns a list of the tokenized sentences
        Args:
            uncased: lower cases text
        """
        if self.sents is not None:
            if uncased:
                sentences = []
                for sen in self.sents:
                    sentences.append([t.lower() for t in sen])
                return sentences
            else:
                return self.sents

    def offsets(self) -> List[List[int]]:
        """
        Returns a list of [start, end) character
        offsets of each token.
        """
        return [t[self.SPAN] for t in self.data]

    def pos(self) -> Any:
        """
        Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self) -> Any:
        """
        Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self) -> Any:
        """
        Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n: int = 1, uncased: bool = False,
               filter_fn: Any = None,
               as_strings: bool = True) -> Any:
        """
        Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_strings: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self) -> Any:
        """
        Group consecutive entity tokens
        with the same NER tag.
        """
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            if ner_tag != non_ent:
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class SpacyTokenizer(object):

    def __init__(self, **kwargs) -> None:
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.nlp = spacy.load(model)
        if not any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            self.nlp.remove_pipe('tagger')
        if 'ner' not in self.annotators:
            self.nlp.remove_pipe('ner')

    def tokenize(self, text: str) -> Tokens:
        clean_text = text.replace('\n', ' ')
        tokens = self.nlp(clean_text)

        sentences = [s for s in tokens.sents]
        sents = []
        for s in sentences:
            sents.append([t.text for t in s])

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                tokens[i].text[0] if len(tokens[i].text) > 0 else '',
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        return Tokens(data, self.annotators, opts={'non_ent': ''}, sents=sents)

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
