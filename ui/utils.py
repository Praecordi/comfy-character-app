import constants


def make_character_description(character):
    if character == "Custom":
        return ""
    else:
        char_key = character.lower()
        char_dict = constants.characters[char_key]

        desc = "\n".join([f"- **{{{key}}}**" for key in char_dict.keys()])

        return desc


def make_key(name):
    return name.replace(" ", "_").lower()


def make_name(key):
    return key.replace("_", " ").title()
