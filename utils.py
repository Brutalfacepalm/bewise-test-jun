import re
import argparse
import sys

import numpy as np
import pandas as pd
import torch
from models import get_matcher, get_yargy_name_parser, get_yargy_company_parser

pd.options.mode.chained_assignment = None


def get_argparse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input_file', dest='input_file',
                        type=str, default='test_data.csv', nargs='?',
                        help='Name of file with dialogs from parse process. Need format CSV. Default: test_date.csv')
    parser.add_argument('-o', '--output_file', dest='output_file',
                        type=str, default=None, nargs='?',
                        help='Name of output file with results of parse if you want. '
                             'If not specify - will be print on screen. '
                             'Default: None')
    parser.add_argument('-gfk', '--greet_fare_k', dest='gfk',
                        type=int, default=3, nargs='?',
                        help='Default: 3 first and last lines accordingly for search greetings and farewells.')
    parser.add_argument('-nk', '--names_k', dest='nk',
                        type=int, default=10, nargs='?',
                        help='Default: 10 first for search names manager and company.')
    parser.add_argument('-ng_min', '--ngrams_min', dest='ng_min',
                        type=int, default=2, nargs='?',
                        help='Default: 2 for min n-grams if possible.')
    parser.add_argument('-ng_max', '--ngrams_max', dest='ng_max',
                        type=int, default=3, nargs='?',
                        help='Default: 3 for max n-grams if possible.')
    return parser


def get_greetings_vocab(path_greetings='greetings.txt'):
    with open(path_greetings) as f:
        greetings = {
            'greeting': list(set([' '.join(re.findall(r'\w+-*\w*', w.strip().lower())) for w in f.readlines()]))}
    return greetings


def get_farewells_vocab(path_farewells='farewells.txt'):
    with open(path_farewells) as f:
        farewells = {
            'farewell': list(set([' '.join(re.findall(r'\w+-*\w*', w.strip().lower())) for w in f.readlines()]))}
    return farewells


def simple_softmax(x):
    return np.exp(x)/sum(np.exp(x))


def get_ngramm(text, min_n=1, max_n=3):
    text = text.split()
    ngrams = []
    for n in range(min_n, max_n+1):
        if len(text) <= n:
            ngrams.append(' '.join(text))
            break
        for w_idx in range(len(text)-n+1):
            ngramma = text[w_idx]
            for i_n in range(1, n):
                ngramma = ' '.join([ngramma, text[w_idx+i_n]])
            ngrams.append(ngramma)
    return ngrams


def match_ngram(sentence_ngram, sentence_match, matcher, matcher_tokenizer, show_phrases=False):
    matches = {}
    for ngram in sentence_ngram:
        text1 = ' '.join(re.findall(r'[а-яА-Я]+-*[а-яА-Я]*', ngram.strip().lower()))
        lengths_word = [len(w) for w in text1.split()]
        if text1 == '' or len(text1) < 4 or np.mean(lengths_word) < 4 or np.max(lengths_word) < 4:
            continue
        matches[text1] = []
        for text2 in sentence_match:
            with torch.inference_mode():
                out = matcher(**matcher_tokenizer(text1, text2, return_tensors='pt').to(matcher.device))
                proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
            matches[text1].append(proba[0])
            if show_phrases:
                print(f'{text1} - {text2} - {proba[0]:.2}')
        matches[text1] = np.quantile(matches[text1], 0.85)

    return np.mean(list(matches.values())) if matches else 0


def matching(replics, feature, matcher, matcher_tokenizer, k_rep=3, ngrmas=(2, 3)):
    if k_rep > replics.shape[0]:
        k_rep = replics.shape[0]
    feature_name, feature = list(feature.items())[0]
    replics[feature_name] = np.zeros(replics.shape[0])
    if feature_name == 'greeting':
        replics_slice = replics['text'].iloc[:k_rep]
    elif feature_name == 'farewell':
        replics_slice = replics['text'].iloc[-k_rep:]
    else:
        replics_slice = replics['text']

    replics_slice_ngrams = replics_slice.apply(lambda x: get_ngramm(x, *ngrmas))
    replics_slice_ngrams_matches = replics_slice_ngrams.apply(lambda x: match_ngram(x, feature,
                                                                                    matcher,
                                                                                    matcher_tokenizer))

    if feature_name == 'greeting':
        replics[feature_name].iloc[:k_rep] = replics_slice_ngrams_matches
    elif feature_name == 'farewell':
        replics[feature_name].iloc[-k_rep:] = replics_slice_ngrams_matches

    if (replics[feature_name] > 0.5).any():
        return np.argmax(replics[feature_name]), replics[feature_name]
    else:
        return 'no match', np.zeros(replics.shape[0])


def get_name(replics, parser, k_rep=10):
    if k_rep > replics.shape[0]:
        k_rep = replics.shape[0]
    replics['name_manager'] = ''
    replics_slice = replics['text'].iloc[:k_rep].values
    find_names = ['' for _ in range(k_rep)]

    for idx_repl, repl in enumerate(replics_slice):
        find_name = parser.findall(repl.title())
        for i in find_name:
            for t in i.tokens:
                if 'Name' in t.forms[0].grams:
                    find_names[idx_repl] = t.forms[0].normalized.title()

    replics['name_manager'].iloc[:k_rep] = find_names
    return [i for i, v in enumerate(replics['name_manager'].astype(bool).astype(int)) if v], replics['name_manager']


def get_company(replics, parser_company, k_rep=10):
    if k_rep > replics.shape[0]:
        k_rep = replics.shape[0]
    replics['name_company'] = ''
    replics_slice = replics['text'].iloc[:k_rep].values
    find_companies = ['' for _ in range(k_rep)]
    tag_company = list(parser_company.rule.productions[0].terms[0].value)

    for idx_repl, repl in enumerate(replics_slice):

        find_company = parser_company.findall(repl)

        for match in find_company:
            name_company = ' '.join([x.value for x in match.tokens if \
                                     [x_norm.forms[0].normalized for x_norm in parser_company.tokenizer(x.value)][0] \
                                     not in tag_company]).title()
            if name_company:
                find_companies[idx_repl] = name_company

    replics['name_company'].iloc[:k_rep] = find_companies
    return [i for i, v in enumerate(replics['name_company'].astype(bool).astype(int)) if v], replics['name_company']


def check_total_dialogies(result):
    for idx_dialog, res_dialog in result.items():
        check_dialog = {'greeting': 'no',
                        'farewell': 'no',
                        'name_manager': 'no',
                        'name_company': 'no'}
        if res_dialog['greeting'][1]:
            check_dialog['greeting'] = 'yes'
        if res_dialog['farewell'][1]:
            check_dialog['farewell'] = 'yes'
        if res_dialog['name_manager'][0]:
            check_dialog['name_manager'] = 'yes'
        if res_dialog['name_company'][0]:
            check_dialog['name_company'] = 'yes'
        result[idx_dialog]['check_total_dialogue'] = check_dialog

    return result


def show_result_parse(result_parse, f=sys.stdout):
    for idx_dialog, res_dialog in result_parse.items():
        print(f'Диалог № {idx_dialog}.', file=f)
        if res_dialog['check_total_dialogue']['greeting'] == 'yes':
            print(f'Менеджер поздоровался тут: \"{res_dialog["greeting"][1]}\"', file=f)
        else:
            print(f'Менеджер не поздоровался.', file=f)

        if res_dialog['check_total_dialogue']['name_manager'] == 'yes':
            print(f'Менеджер представился тут: \"{res_dialog["replica_with_name"][0]}\"', file=f)
            print(f'Менеджера зовут: \"{res_dialog["name_manager"][0]}\"', file=f)
        else:
            print('Менеджер не представился.', file=f)

        if res_dialog['check_total_dialogue']['name_company'] == 'yes':
            print(f'Менеджер из компании: \"{res_dialog["name_company"][0]}\"', file=f)
        else:
            print(f'Менеджер не назвал компанию.', file=f)

        if res_dialog['check_total_dialogue']['farewell'] == 'yes':
            print(f'Менеджер попрощался тут: \"{res_dialog["farewell"][1]}\"', file=f)
        else:
            print(f'Менеджер не попрощался.', file=f)

        if res_dialog['check_total_dialogue']['farewell'] == 'yes' and res_dialog['check_total_dialogue']['greeting'] == 'yes':
            print(f'Менеджер поздоровался и попрощался.', file=f)
        else:
            print(f'Менеджер не поздоровался и/или не попрощался.', file=f)


def load_files(args):
    greetings = get_greetings_vocab()
    farewells = get_farewells_vocab()
    dialogs = pd.read_csv(args.input_file)
    dialogs_manager = dialogs[dialogs['role'] == 'manager']

    return greetings, farewells, dialogs_manager


def load_models():
    matcher, matcher_tokenizer = get_matcher()
    name_parser = get_yargy_name_parser()
    company_parser = get_yargy_company_parser()

    return matcher, matcher_tokenizer, name_parser, company_parser


def start_parse(dialogs_manager, greetings, farewells, matcher, matcher_tokenizer, name_parser, company_parser, args):
    # dialogs_by_id = {}
    parse_dialog_result = {}
    gkf, nk = args.gfk, args.nk
    ng_min, ng_max = args.ng_min, args.ng_max
    for dlg_id in dialogs_manager['dlg_id'].unique():
        parse_dialog_result[dlg_id] = {'greeting': [],
                                       'farewell': [],
                                       'replica_with_name': [],
                                       'name_manager': [],
                                       'replica_with_company': [],
                                       'name_company': [],
                                       'check_total_dialogue': {}}
        replics = dialogs_manager[dialogs_manager['dlg_id'] == dlg_id]
        idx_rep, replics['greeting'] = matching(replics.copy(), greetings, matcher, matcher_tokenizer,
                                                k_rep=gkf, ngrmas=(ng_min, ng_max))
        parse_dialog_result[dlg_id]['greeting'] = [replics['greeting'].iloc[idx_rep],
                                                   replics['text'].iloc[idx_rep]] if idx_rep != 'no match' else \
                                                    [None, None]

        idx_rep, replics['farewell'] = matching(replics.copy(), farewells, matcher, matcher_tokenizer,
                                                k_rep=gkf, ngrmas=(ng_min, ng_max))
        parse_dialog_result[dlg_id]['farewell'] = [replics['farewell'].iloc[idx_rep],
                                                   replics['text'].iloc[idx_rep]] if idx_rep != 'no match' else \
                                                    [None, None]

        idx_name, replics['name_manager'] = get_name(replics.copy(), name_parser, k_rep=nk)
        if idx_name:
            for idx in idx_name:
                parse_dialog_result[dlg_id]['replica_with_name'].append(replics.iloc[idx]['text'])
                parse_dialog_result[dlg_id]['name_manager'].append(replics.iloc[idx]['name_manager'])
        else:
            parse_dialog_result[dlg_id]['replica_with_name'].append(None)
            parse_dialog_result[dlg_id]['name_manager'].append(None)

        idx_company, replics['name_company'] = get_company(replics.copy(), company_parser, k_rep=nk)
        if idx_company:
            for idx in idx_company:
                parse_dialog_result[dlg_id]['replica_with_company'].append(replics.iloc[idx]['text'])
                parse_dialog_result[dlg_id]['name_company'].append(replics.iloc[idx]['name_company'])
        else:
            parse_dialog_result[dlg_id]['replica_with_company'].append(None)
            parse_dialog_result[dlg_id]['name_company'].append(None)

        # dialogs_by_id[dlg_id] = replics
    parse_dialog_result = check_total_dialogies(parse_dialog_result)
    return parse_dialog_result


if __name__ == "__main__":
    pass