import inflect

inflect_p = inflect.engine()


def stem_func(word):
    sing_w = inflect_p.singular_noun(word)
    if not sing_w:
        return word
    return sing_w

