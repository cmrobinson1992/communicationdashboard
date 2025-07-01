import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
try:
    sent_detector = nltk.tokenize.PunktTokenizer()
except LookupError:
    nltk.download("punkt_tab")
class Preprocess(object):
    def __init__(self):
        self.regex = '^\s+|\W+|[0-9]|\s+$' # for autograder consistency, please do not change

    def clean_text(self, text):
        '''
        Clean the input text string:
            1. Remove HTML formatting (use Beautiful Soup)
            2. Remove non-alphabet characters such as punctuation or numbers and replace with ' '
               You may refer back to the slides for this part (For autograder consistency, 
               we implement this part for you, please do not change it.)
            3. Remove leading or trailing white spaces including any newline characters
            4. Convert to lower case
            5. Tokenize and remove stopwords using nltk's 'english' vocabulary
            6. Rejoin remaining text into one string using " " as the word separator
            
        Args:
            text: string 
        
        Return:
            cleaned_text: string
        '''

        soup = BeautifulSoup(text, 'html.parser')
        cleaned_text = soup.get_text()
        # Step 2 is implemented for you, please do not change
        cleaned_text = re.sub(self.regex,' ',cleaned_text).strip()
        # Step 3
        cleaned_text = cleaned_text.strip()
        # Step 4
        cleaned_text = cleaned_text.lower()
        # Step 5
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(cleaned_text)
        cleaned_text = [w for w in word_tokens if not w.lower() in stop_words]
        # Step 6
        cleaned_text = ' '.join(cleaned_text)
        return cleaned_text
   #     raise NotImplementedError

    def clean_dataset(self, data):
        '''
        Given an array of strings, clean each string in the array by calling clean_text()
            
        Args:
            data: list of N strings
        
        Return:
            cleaned_data: list of cleaned N strings
        '''
        cleaned_data = [self.clean_text(x) for x in data]
        return cleaned_data
        raise NotImplementedError


# Note that clean_wos is outside of the Preprocess class
def clean_wos(x_train, x_test):
    '''
    ToDo: Clean both the x_train and x_test dataset using clean_dataset from Preprocess
    
    Input:
        x_train: list of N strings
        x_test: list of M strings
        
    Output:
        cleaned_text_wos: list of cleaned N strings
        cleaned_text_wos_test: list of cleaned M strings
    '''
    preprocess = Preprocess()
    return preprocess.clean_dataset(x_train), preprocess.clean_dataset(x_test)
    raise NotImplementedError
