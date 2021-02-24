from sys import setswitchinterval
from typing import Match
import utils
import math
import nltk
# tagger
nltk.download('averaged_perceptron_tagger')
# stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
# stemmer
nltk.download('punkt')
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()


def isNounOrVerb(token_tag: str) -> bool:
    '''
    Returns true if the given token tag corresponds to any type of noun or verb, returns false otherwise.
    '''
    return token_tag.startsWith('NN') or token_tag.statsWith('VB')


# see: https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/
def preprocess(req_text: str, match_type: int):
    '''
    Preprocesses requirement text though tokenisation, tagging, stop-word removal, and stemming.
    Returns all resulting tokens by default.
    Retrusn all verb and/or noun token pairs for matcher 3.
    '''
    # split text into words
    tokens = nltk.word_tokenize(req_text)
    # tag words
    tagged_tokens = nltk.pos_tag(tokens)
    # remove non-alphanumerical signs
    tagged_tokens = filter(lambda tagged_token: tagged_token[0].isalpha(), tagged_tokens)
    # make tokens to lower case
    tagged_tokens = map(lambda tagged_token: (tagged_token[0].lower(), tagged_token[1]), tagged_tokens)
    # remove stop-word tokens
    tagged_tokens = filter(lambda tagged_token: not tagged_token[0] in stop_words, tagged_tokens)
    # stem tokens
    tagged_tokens = map(lambda tagged_token: (porter_stemmer.stem(tagged_token[0]), tagged_token[1]), tagged_tokens)

    # accumulate result
    tagged_tokens = list(tagged_tokens)

    # further processing, depending on the matcher
    tokens = []
    if match_type == 3:
        # iterate over all tagged token pairs
        for tagged_token1, tagged_token2 in zip(tagged_tokens, tagged_tokens[1:]):
            token1, tag1 = tagged_token1
            token2, tag2 = tagged_token2
            # append all verb and/or noun combinations
            if isNounOrVerb(tag1) and isNounOrVerb(tag2):
                tokens.append(token1 + token2)
    else:
        # take all tokens
        tokens = list(map(lambda tagged_token: tagged_token[0], tagged_tokens))
    
    # return resulting tokens
    return tokens


def cosine_similarity(v1, v2):
    '''
    Computes the cosine similarity between v1 and v2.

    '''
    sum_xx, sum_yy, sum_xy = 0, 0, 0
    for (x, y) in zip(v1, v2):
        sum_xx += x * x
        sum_yy += y * y
        sum_xy += x * y
    return sum_xy / math.sqrt(sum_xx * sum_yy)


def link_requirements(high_req_ids, low_req_ids, sim_matrix, match_type):
    '''
    Tries to link each high-level requirement with each low-level requirements, based on their similarity.
    Filtering of the links is done based on the match type.
    Returns a list, of which each entry contains the high-level requirements id, and a list of the low-level requirement ids linked to it.
    '''
    result = []
    for h, high_req_id in enumerate(high_req_ids):
        links = []
        for l, low_req_id in enumerate(low_req_ids):
            # obtain similarity
            sim = sim_matrix[h][l]
            max_sim = max(sim_matrix[h])
            # determine if there is a link
            is_link = any([
                match_type == 0 and sim > 0,
                match_type == 1 and sim >= 0.25,
                match_type == 2 and sim >= 0.67 * max_sim,
                match_type == 3 and sim > 0 and sim >= 0.67 * max_sim
            ])
            # if so, add the link
            if is_link:
                links.append(low_req_id)
        result.append([high_req_id, ",".join(links)])
    return result


def process(match_type):
    # read low-level and high-level requirements
    low_reqs = utils.get_file_rows("/input/low.csv")
    high_reqs = utils.get_file_rows("/input/high.csv")

    # preprocess low-level and high-level requirements
    processed_low_reqs, processed_high_reqs = [], []
    for req_id, req_text in low_reqs:
        tokens = preprocess(req_text, match_type)
        processed_low_reqs.append([req_id, tokens])
    for req_id, req_text in high_reqs:
        tokens = preprocess(req_text, match_type)
        processed_high_reqs.append([req_id, tokens])
    
    # put all processed requirements into a single list
    all_processed_reqs = processed_low_reqs + processed_high_reqs
    num_reqs = len(all_processed_reqs)
    # put all tokens into a single list
    all_tokens = []
    for _, req_tokens in all_processed_reqs:
        all_tokens.extend(req_tokens)
    # construct master vocabulary from a duplicate-free set of all tokens
    master_vocab = set(all_tokens)

    # determine for each master vocabulary token in how many requirements it is present
    master_token_freq = dict()
    for master_token in master_vocab:
        master_token_freq[master_token] = 0
        for _, req_tokens in all_processed_reqs:
            if master_token in req_tokens:
                master_token_freq[master_token] += 1
    
    # calculate the vector representation of the requirements
    req_vectors = dict()
    for req_id, req_tokens in all_processed_reqs:
        req_vectors[req_id] = []
        for master_token in master_vocab:
            if master_token in req_tokens:
                tf = req_tokens.count(master_token)
                idf = math.log2(num_reqs / master_token_freq[master_token])
                req_vectors[req_id].append(tf * idf)
            else:
                req_vectors[req_id].append(0)
    
    # obtain low-level and high-level requirement ids
    high_req_ids = list(map(lambda high_req: high_req[0], high_reqs))
    low_req_ids = list(map(lambda low_req: low_req[0], low_reqs))
    
    # generate the similarity matrix
    sim_matrix = []
    for h, high_req_id in enumerate(high_req_ids):
        sim_matrix.append([])
        for l, low_req_id in enumerate(low_req_ids):
            sim = cosine_similarity(req_vectors[high_req_id], req_vectors[low_req_id])
            sim_matrix[h].append(sim)

    # link the requirements based on the chosen matcher
    rows = link_requirements(high_req_ids, low_req_ids, sim_matrix, match_type)

    # write the result to output
    utils.write_file("/output/links.csv", ["id", "links"], rows)
