import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from yargy import Parser, rule
from yargy.predicates import gram, dictionary


def get_matcher():
    matcher_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    matcher_tokenizer = AutoTokenizer.from_pretrained(matcher_checkpoint)
    matcher = AutoModelForSequenceClassification.from_pretrained(matcher_checkpoint)
    if torch.cuda.is_available():
        matcher.cuda()
    return matcher, matcher_tokenizer


def get_yargy_name_parser():
    G = gram('Name')
    R_name = rule(dictionary({'меня', 'зовут', 'я', 'это', 'мое', 'имя', 'вас', 'беспокоит'}).repeatable(),
                  G.repeatable())
    parser = Parser(R_name)

    return parser


def get_yargy_company_parser():
    G = gram('NOUN')
    R_company = rule(dictionary({'компания', 'фирма', 'организация',
                                 'предприятие', 'контора', 'ООО',
                                 'ЗАО', 'ОАО', 'АО', 'ПАО', 'ИП'}), G.repeatable())
    parser = Parser(R_company)

    return parser


if __name__ == "__main__":
    pass