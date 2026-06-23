import json
import random



#Introduces errors that are common to see during ocr operations

CHAR_CONFUSIONS = {
    'O': ['0'], '0': ['O'],
    'I': ['1', 'l'], '1': ['I', 'l'], 'l': ['I', '1'],
    'S': ['5'], '5': ['S'],
    'B': ['8'], '8': ['B'],
    'Z': ['2'], '2': ['Z'],
    'G': ['6'], '6': ['G'],
    'A': ['4'], '4': ['A'],
    'e': ['c'], 'c': ['e'],
    'n': ['m'], 'm': ['n', 'rn'],
}

def char_substitution(text, prob=0.08):
    result = []
    for ch in text:
        if ch in CHAR_CONFUSIONS and random.random() < prob:
            result.append(random.choice(CHAR_CONFUSIONS[ch]))
        else:
            result.append(ch)
    return ''.join(result)

def random_char_drop(text, prob=0.015):
    return ''.join(ch for ch in text if random.random() > prob)

def random_char_insert(text, prob=0.01):
    noise_chars = ['.', ',', "'", '`', '-', '_']
    result = []
    for ch in text:
        result.append(ch)
        if random.random() < prob:
            result.append(random.choice(noise_chars))
    return ''.join(result)

def word_spacing_corruption(text, prob=0.04):
    words = text.split(' ')
    result = []
    i = 0
    while i < len(words):
        if random.random() < prob and i < len(words) - 1:
            result.append(words[i] + words[i+1])
            i += 2
        elif random.random() < prob / 2 and len(words[i]) > 4:
            split_point = random.randint(1, len(words[i]) - 1)
            result.append(words[i][:split_point])
            result.append(words[i][split_point:])
            i += 1
        else:
            result.append(words[i])
            i += 1
    return ' '.join(result)

def random_whitespace_injection(text, prob=0.05):
    result = []
    for ch in text:
        result.append(ch)
        if ch == ' ' and random.random() < prob:
            result.append(' ' * random.randint(1, 3))
    return ''.join(result)

def case_corruption(text, prob=0.02):
    result = []
    for ch in text:
        if random.random() < prob and ch.isalpha():
            result.append(ch.swapcase())
        else:
            result.append(ch)
    return ''.join(result)

def line_break_corruption(text, prob=0.3):
    lines = text.split('\n')
    result = []
    for line in lines:
        result.append(line)
        if random.random() < prob:
            result.append('')
    return '\n'.join(result)

def apply_random_noise(text, intensity='medium'):
    intensity_multiplier = {'light': 0.5, 'medium': 1.0, 'heavy': 1.8}[intensity]

    noise_functions = [
        (char_substitution, 0.08),
        (random_char_drop, 0.015),
        (random_char_insert, 0.01),
        (word_spacing_corruption, 0.04),
        (random_whitespace_injection, 0.05),
        (case_corruption, 0.02),
    ]

    num_to_apply = random.randint(2, len(noise_functions))
    chosen = random.sample(noise_functions, num_to_apply)

    for func, base_prob in chosen:
        adjusted_prob = base_prob * intensity_multiplier * random.uniform(0.7, 1.3)
        text = func(text, prob=adjusted_prob)

    if random.random() < 0.2 * intensity_multiplier:
        text = line_break_corruption(text, prob=0.3 * intensity_multiplier)

    return text



INTENSITY_DISTRIBUTION = ['light'] * 3 + ['medium'] * 5 + ['heavy'] * 2

random.seed(42)  # reproducible results

input_file = "merged_clean.jsonl"
output_file = "combined_noisy.jsonl"

count = 0
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:

    for line in infile:
        record = json.loads(line)
        clean_text = record["clean_text"]

        intensity = random.choice(INTENSITY_DISTRIBUTION)
        noisy_text = apply_random_noise(clean_text, intensity=intensity)

        outfile.write(json.dumps({
            "source": record["source"],
            "clean_text": clean_text,
            "noisy_text": noisy_text,
            "intensity": intensity
        }) + "\n")
        count += 1

print(f"Noise injection complete: {count} records saved to {output_file}")