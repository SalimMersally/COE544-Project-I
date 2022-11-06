
from spellchecker import SpellChecker

def correct_word(str):
    

    spell = SpellChecker()

    # find those words that may be misspelled
    misspelled = spell.unknown(str.split())
    corrected = []

    for word in misspelled:
        # Get the one `most likely` answer
        #print(spell.correction(word))

        # Get a list of `likely` options
        #print(spell.candidates(word))
    
        corrected.append(spell.correction(word))
    
        return(" ".join(corrected[::-1]))



print(correct_word("that"))
    
