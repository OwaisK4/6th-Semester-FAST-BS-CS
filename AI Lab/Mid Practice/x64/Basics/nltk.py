# Code1: 
import nltk
nltk.download('punkt')

# Code2:
#Removing stop words with NLTK in Python
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
print(stopwords.words('english'))
#The following program removes stop words from a piece of text:
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
example_sent = """This is a sample sentence,
showing off the stop words filtration."""
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sent)
# converts the words in word_tokens to lower case and then checks whether
#they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#with no lower case conversion
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
    
    print(word_tokens)
    print(filtered_sentence)

#
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

sentences = "My name is Muhammad Ahmed. My roll no is 3161. I study in FAST NUCES."
print("Original string:\n",sentences)

sentence_tokens = sent_tokenize(sentences)
print("\n[TASK 1] Sentence-tokenized copy in the list:\n", sentence_tokens)

word_tokens = word_tokenize(sentences)
print("\n[TASK 2] List of words:\n", word_tokens)

print("\n[TASK 3] Read the list:")
formatted_sentences = [word_tokenize(sentence) for sentence in sentence_tokens]
for formatted_sentence in formatted_sentences:
    print(formatted_sentence)

print("\nRead the list:")
for sentence in sentence_tokens:
    print(sentence)