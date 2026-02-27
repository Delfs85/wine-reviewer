import requests, os
from dotenv import load_dotenv
load_dotenv()

r = requests.get(
    'https://www.cellartracker.com/xlquery.asp',
    params={
        'User': os.getenv('CELLARTRACKER_USER'),
        'Password': os.getenv('CELLARTRACKER_PASS'),
        'Format': 'Tab',
        'Table': 'Notes',
        'Wine': 'Frederic Cossard Bedeau Rouge 2020'
    },
    timeout=10
)
lines = r.text.strip().split('\n')
print(f'Total lines: {len(lines)}')
for line in lines[1:5]:
    parts = line.split('\t')
    print(f'Wine: {parts[5] if len(parts) > 5 else "?"}')
