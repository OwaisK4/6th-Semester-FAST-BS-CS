# Code1:
import spacy
nlp = spacy.load('en_core_web_sm')
sentence = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(sentence)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

# Code2:
# First we need to import spacy
import spacy
# Creating blank language object then
# tokenizing words of the sentence
nlp = spacy.blank("en")
doc = nlp("GeeksforGeeks is a one stop\
learning destination for geeks.")
for token in doc:
    print(token)

# Code3:
#Here is an example to show what other functionalities can be enhanced by adding modules to the
# pipeline.
import spacy
# loading modules to the pipeline.
nlp = spacy.load("en_core_web_sm")
# Initialising doc with a sentence.
doc = nlp("If you want to be an excellent programmer \
, be consistent to practice daily on GFG.")
# Using properties of token i.e. Part of Speech and Lemmatization
for token in doc:
    print(token, " | ",
    spacy.explain(token.pos_)," | ", token.lemma_)

#
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
sentences = "My name is . I study in FAST NUCES."
doc = nlp(sentences)

for sentence_tokens in doc:
    print(sentence_tokens.text)

displacy.render(doc, style="dep", options={'distance': 90}, jupyter=True)