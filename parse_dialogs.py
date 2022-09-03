from utils import get_argparse, load_files, load_models, start_parse, show_result_parse

"""
Parse file with dialogs and print results on screen or to file.
Script look for manager greeting, name of manager, name of company and words farewell by manager.
Script return replicas of manager and check that manager say hello and bye.
"""

if __name__ == "__main__":
    parser = get_argparse()
    args = parser.parse_args()

    greetings, farewells, dialogs_manager = load_files(args)
    matcher, matcher_tokenizer, name_parser, company_parser = load_models()

    parse_dialog_result = start_parse(dialogs_manager, greetings, farewells,
                                      matcher, matcher_tokenizer,
                                      name_parser, company_parser, args)
    if args.output_file is None:
        show_result_parse(parse_dialog_result)
    else:
        with open(args.output_file, 'w+') as f:
            show_result_parse(parse_dialog_result, f)
