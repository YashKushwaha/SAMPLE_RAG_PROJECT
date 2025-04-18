def load_text(text_file):
    with open(text_file, 'r') as f:
        text = f.read()
    return text

def clean_text(text):
    lines = [i for i in text.split('\n') if (len(i) > 20) and len(i.split() > 4)]
    return '\n'.join(lines)
 
