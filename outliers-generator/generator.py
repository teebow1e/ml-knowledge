import random
import string

def generate_length_overemphasis():
    """Generate a long password with low complexity."""
    length = random.randint(18, 24)
    printable_characters = string.printable[:62] + "!@#$%&*_-+=?"
    first_char = random.choice(printable_characters)
    last_char = random.choice(printable_characters)
    middle_chars = first_char * (length - 2)
    return middle_chars + last_char

def generate_common_pattern():
    """Generate a password with a common pattern."""
    patterns = ["123456", "abcdef", "password", "letmein", "welcome", "qwerty"]
    return (random.choice(patterns))*int((random.choice([1,2,3])))

def generate_simple_repeats(length=12):
    """Generate a password with simple repeated sequences."""
    repeat_pattern = random.choice(['ab', '01', 'xy', 'qw', 'we', '12', '34', '56', '78', '90', 'cd', 'ef', 'gh', 'ij', 'kl', 'mn', 'op', 'rs', 'tu', 'vw', 'xz', 'rt', 'ty', 'yu', 'ui', 'io', 'op', 'as', 'df', 'gh', 'jk', 'lz', 'xc', 'vb', 'nm'])
    return (repeat_pattern * (length // 2))[:length]

# biased_passwords = [
#     generate_length_overemphasis(),
#     generate_common_pattern(),
#     generate_character_set_bias(),
#     generate_keyboard_pattern(),
#     generate_simple_repeats()
# ]

# for pwd in biased_passwords:
#     print(pwd)

number_of_passwords = 10000
all_biased_passwords = []

for _ in range(number_of_passwords // 5):
    all_biased_passwords.extend([
        generate_length_overemphasis(),
        generate_common_pattern(),
        generate_simple_repeats()
    ])

for pwd in all_biased_passwords:
    print(pwd)
