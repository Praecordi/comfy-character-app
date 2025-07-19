import constants


def make_character_description(character):
    char_key = make_key(character)
    if char_key == "custom" or char_key not in constants.characters:
        return ""
    else:
        char_dict = constants.characters[char_key]

        desc = "\n".join([f"- **{{{key}}}**" for key in char_dict.keys()])

        return desc


def make_key(name):
    return name.replace(" ", "_").lower()


def make_name(key):
    return key.replace("_", " ").title()
