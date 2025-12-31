import unittest
import nltk
from cli.text_processing import (
    make_case_insensitive,
    remove_punctuation,
    tokenize,
    remove_stopwords,
    stem
)

class TestTextProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure necessary NLTK data is downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def test_make_case_insensitive(self):
        self.assertEqual(make_case_insensitive("Hello WORLD"), "hello world")
        self.assertEqual(make_case_insensitive("python"), "python")

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation("hello, world!"), "hello world")
        self.assertEqual(remove_punctuation("No-punctuation here."), "Nopunctuation here")

    def test_tokenize(self):
        self.assertEqual(tokenize("hello world"), ["hello", "world"])
        self.assertEqual(tokenize("  multiple   spaces  "), ["multiple", "spaces"])
        self.assertEqual(tokenize(""), [])

    def test_remove_stopwords(self):
        tokens = ["this", "is", "a", "test", "of", "the", "system"]
        # 'this', 'is', 'a', 'of', 'the' are common stopwords
        filtered = remove_stopwords(tokens)
        self.assertNotIn("this", filtered)
        self.assertNotIn("is", filtered)
        self.assertIn("test", filtered)
        self.assertIn("system", filtered)

    def test_stem(self):
        tokens = ["running", "runs", "easily", "fairly"]
        stemmed = stem(tokens)
        # PorterStemmer results
        self.assertEqual(stemmed[0], "run")
        self.assertEqual(stemmed[1], "run")
        self.assertEqual(stemmed[2], "easili")
        self.assertEqual(stemmed[3], "fairli")

if __name__ == "__main__":
    unittest.main()

