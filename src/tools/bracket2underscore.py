import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-input", help="input path for parsed data")
parser.add_argument("-output", help="output path for results replacing spaces within [] with _")
args = parser.parse_args()

data = list()
with open(args.output, 'w') as output:
    with open(args.input, 'r') as input:
        for line in input:
            line = line.strip()
            units = []
            concept_indices = []
            current_unit = ''
            within_concept = False
            for ch in line:
                if not within_concept and (ch == " " or ch == "$"):
                    if current_unit != '':
                        units.append(current_unit.replace(' ', '_'))
                        current_unit = ''
                elif ch == '[':
                    concept_indices.append(len(units))
                    within_concept = True
                elif ch == ']':
                    within_concept = False
                else:
                    current_unit += ch
            if current_unit != '':
                units.append(current_unit.replace(' ', '_'))
            if len(units) > 20:
                output.write(' '.join(units))
                output.write('\n')
