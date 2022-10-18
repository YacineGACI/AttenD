import json, re

def find_different_word(context, sentence):
    return re.findall(context.lower().replace('.', '\.').replace('(', '\(').replace(')', '\)').replace('blank', '(.*)'), sentence.lower())[0]


if __name__ == "__main__":

    input_filename = "data/evaluation/stereoset/stereoset-dev.json"
    output_filename = "data/evaluation/stereoset/stereoset_processed.tsv"


    with open(input_filename, "r") as f:
        with open(output_filename, 'w') as wf:
            data = json.load(f)
            intrasentences = data["data"]["intrasentence"]
            for intrasentence in intrasentences:
                context = intrasentence['context']
                
                for s in intrasentence['sentences']:
                    if s['gold_label'] == 'unrelated':
                        unrelated = find_different_word(context, s["sentence"])
                    if s['gold_label'] == 'stereotype':
                        stereotype = find_different_word(context, s["sentence"])
                    if s['gold_label'] == 'anti-stereotype':
                        anti_stereotype = find_different_word(context, s["sentence"])

                if context.count("BLANK") > 1:
                    continue
                
                wf.write("{}\t{}\t{}\t{}\t{}\n".format(context.replace("BLANK", "[MASK]"), intrasentence["bias_type"], unrelated, stereotype, anti_stereotype))
                

    