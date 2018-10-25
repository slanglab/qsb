import pyparsing

def jdoc_to_constit_list(jdoc):
    '''input: jdoc, output: constit_parse as list'''
    parse = jdoc["parse"].replace("\n", "")
    parens = pyparsing.nestedExpr('(', ')')
    parse = parens.parseString(parse).asList()
    return parse
