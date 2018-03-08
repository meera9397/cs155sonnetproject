########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 HMM helper
########################################

import re
import numpy as np
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import animation
from matplotlib.animation import FuncAnimation


####################
# WORDCLOUD FUNCTIONS
####################

def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r:d-r, -r:d-r]
    circle = (x**2 + y**2 <= r**2)

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask

def text_to_wordcloud(text, max_words=50, title='', show=True):
    plt.close('all')

    # Generate a wordcloud image.
    wordcloud = WordCloud(random_state=0,
                          max_words=max_words,
                          background_color='white',
                          mask=mask()).generate(text)

    # Show the image.
    if show:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=24)
        plt.show()

    return wordcloud

def states_to_wordclouds(hmm, obs_map, syllable_map, n_syllables, n_lines,show=True):
    # Initialize.
    M = 100000
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = []

    # Generate a large emission.
    emission, states = hmm.generate_emission(obs_map_r, syllable_map, n_syllables, n_lines)
    emission = [item for sublist in emission for item in sublist] # flatten list
    
    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    n_states_copy = n_states
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        if (len(obs_lst) == 0):
            n_states_copy -= 1
            continue
        obs_count.append(obs_lst)

    # For each state, convert it into a wordcloud.
    for i in range(n_states_copy):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = ' '.join(sentence)
        max_words = 50
        wordclouds.append(text_to_wordcloud(sentence_str, max_words=max_words, title='State %d' % i, show=show))

    return wordclouds


####################
# HMM FUNCTIONS
####################

def parse_syllables(text):
    ''' Creates a dictionary of [key = word, value = # syllables] '''
    syllable_map = {}
    lines = text.split('\n')
    
    for line in lines:
        if len(line) == 0:
            continue
        line = line.split()
        word = line[0]
        syllable_count = line[-1]
        if len(syllable_count) == 2:
            syllable_count = line[-2]
        syllable_count = int(syllable_count)
        syllable_map[word] = syllable_count
        
    return syllable_map

def parse_backwards_observations(text):
    ''' Parses the text for backwards HMM generation '''
    
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        line = line[::-1]
        obs_elem = []
        
        for word in line:
            word = re.sub("\d+", "", word)
            if (word == ""):
                continue
            word = re.sub(r'[^-\w\']', '', word).lower()
                
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            
            # Add the encoded word.
            obs_elem.append(obs_map[word])
        
        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map
    
def parse_observations(text):
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []
        
        for word in line:
            word = re.sub("\d+", "", word)
            if (word == ""):
                continue
            word = re.sub(r'[^-\w\']', '', word).lower()
                
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            
            # Add the encoded word.
            obs_elem.append(obs_map[word])
        
        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map

def obs_map_reverser(obs_map):
    obs_map_r = {}

    for key in obs_map:
        obs_map_r[obs_map[key]] = key

    return obs_map_r


def sample_backwards_sonnet(hmm, obs_map, syllable_map, n_syllables, n_lines, rhymes):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.rhyming_emission(obs_map, obs_map_r, syllable_map, n_syllables, n_lines, rhymes)
    
    # punctuation list
    punctuation_list = [',','.',':','?']
    punctuation_probs = [0.6, 0.1, 0.2, 0.1]

    sonnet_string = ""
    for line in range(n_lines):
        punctuation = np.random.choice(punctuation_list, p = punctuation_probs)
        if (line == n_lines - 1):
            punctuation = '.'
        sonnet = [obs_map_r[i] for i in emission[line]]
        sonnet = sonnet[::-1]
        sonnet_string += str((' '.join(sonnet).capitalize()) + punctuation + '\n')
    
    return sonnet_string
    
    
def sample_sonnet(hmm, obs_map, syllable_map, n_syllables, n_lines):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(obs_map_r, syllable_map, n_syllables, n_lines)
            
    # punctuation list
    punctuation_list = [',','.',':','?']
    punctuation_probs = [0.6, 0.1, 0.2, 0.1]

    sonnet_string = ""
    for line in range(n_lines):
        punctuation = np.random.choice(punctuation_list, p = punctuation_probs)
        sonnet = [obs_map_r[i] for i in emission[line]]
        if (line == n_lines - 1):
            punctuation = '.'
        sonnet_string += str((' '.join(sonnet).capitalize()) + punctuation + '\n')
       
    
    return sonnet_string

####################
# HMM VISUALIZATION FUNCTIONS
####################

def visualize_sparsities(hmm, O_max_cols=50, O_vmax=0.1):
    plt.close('all')
    plt.set_cmap('viridis')

    # Visualize sparsity of A.
    plt.imshow(hmm.A, vmax=1.0)
    plt.colorbar()
    plt.title('Sparsity of A matrix')
    plt.show()

    # Visualize parsity of O.
    plt.imshow(np.array(hmm.O)[:, :O_max_cols], vmax=O_vmax, aspect='auto')
    plt.colorbar()
    plt.title('Sparsity of O matrix')
    plt.show()



