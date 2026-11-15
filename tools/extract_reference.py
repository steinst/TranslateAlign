"""Turn the verbose log of the original Bleualign (-v 2, run as a black box) into JSON: per article the 1-1 anchors
of the first pass, the final beads, and the counts per alignment type.  Numbers only, no text."""
import json, re, sys
ansi = re.compile(r'\x1b\[[0-9;]*m')
def parse(path):
    arts, cur, mode = [], None, None
    for raw in open(path, encoding='utf-8', errors='replace'):
        line = ansi.sub('', raw.rstrip('\n'))
        if line.startswith('reading in article'):
            cur = {'anchors': [], 'beads': [], 'types': {}, 'summary': []}; arts.append(cur); mode = None
        elif cur is None:
            continue
        elif line.startswith('Results of BLEU 1-to-1 alignment'):
            mode = 'anchors'
        elif mode == 'anchors' and re.match(r'^\d+: \d+$', line):
            a, b = line.split(': '); cur['anchors'].append([int(a), int(b)])
        elif 'out of' in line and 'aligned' in line:
            cur['summary'].append(line); mode = None
        elif line.startswith('alignment: '):
            s, t = line[len('alignment: '):].split(' - ')
            cur['beads'].append([[int(x) for x in s.split(',') if x], [int(x) for x in t.split(',') if x]])
        else:
            m = re.match(r'precision for alignment type (\w+) \( (\d+) alignment pairs\)', line)
            if m:
                cur['types'][m.group(1)] = int(m.group(2))
    return arts
for name in sys.argv[1:]:
    arts = parse(name + '.log')
    json.dump(arts, open(name + '.reference.json', 'w'))
    print(name, len(arts), 'articles;', sum(len(a['anchors']) for a in arts), 'anchors;', sum(len(a['beads']) for a in arts), 'beads;',
          {k: sum(a['types'].get(k, 0) for a in arts) for k in ('BLEU', 'GAPFILLER', 'GALECHURCH')})
