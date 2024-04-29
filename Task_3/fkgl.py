import re
# Implementing the FKGL formula manually since the initial attempt to use an external library failed

def calculate_fkgl(text):
    # Splitting the text into sentences
    sentences = text.split('.')
    num_sentences = len(sentences) - 1  # Adjusting for the last split that does not result in a sentence
    
    # Splitting the text into words
    words = text.split()
    num_words = len(words)
    
    # Counting syllables in the text
    # A simple approximation: assuming words of three or more letters have one syllable,
    # and each additional three letters add another syllable
    # syllables = sum([1 + (len(word) - 3) // 3 for word in words if len(word) > 2])
    syllables = re.split(r'(?<=\b\w)(?=[aeiou])| ', text, flags=re.IGNORECASE)
    print(syllables)
    # Calculating average sentence length and average syllables per word
    avg_sentence_length = num_words / num_sentences
    avg_syllables_per_word = len(syllables) / num_words
    
    # FKGL formula
    fkgl_score = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    return fkgl_score


text = """
The journey across the ancient forest was both thrilling and perilous, with hidden dangers lurking beneath its canopy.
Students are required to complete their science project by the end of next month, which includes both research and experimentation.
Advances in technology have made solar energy more affordable and accessible to people around the world.
The author's use of symbolism in the novel provides a deep insight into the characters' motivations and desires.
Environmental conservation efforts are crucial for preserving biodiversity and ensuring the health of our planet.
"""

# Recalculate FKGL score using the manual method
fkgl_score_manual = calculate_fkgl(text)
print(fkgl_score_manual)
