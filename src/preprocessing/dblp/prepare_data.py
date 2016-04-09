import argparse
parser = argparse.ArgumentParser()

# Get papers pulished in conferences containing at least one of the following keyphrases
keywords = ["mining", "learning", "database", \
            "search", "retrieval", "intelligence", \
            "computer vision", "image"]

parser.add_argument("-input", help="input path for AMiner-Paper.txt")
parser.add_argument("-output", help="output path for processed paper content")
args = parser.parse_args()

keywords = [i.strip().lower() for i in keywords]

with open(args.output, 'w') as output:
    with open(args.input, 'r') as input:
        title = ""
        abstract = ""
        venue_interested = False
        for line in input:
            if line.strip() == "":
                if venue_interested and (abstract != "" or title != ""):
                    if len(title) > 0:
                        output.write(title)
                        output.write('\n')
                    if len(abstract) > 0:
                        output.write(abstract)
                        output.write('\n')
                title = ""
                abstract = ""
                venue_interested = False
                continue
            if line[0:2] == "#c":
                venue = line[2:].strip().lower()
                for keyword in keywords:
                    if venue.find(keyword) >= 0:
                        venue_interested = True
            if line[0:2] == "#!":
                abstract = line[2:].strip()
            if line[0:2] == "#*":
                title = line[2:].strip()
