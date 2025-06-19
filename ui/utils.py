from constants import characters


def make_character_description(character):
    if character == "Custom":
        return ""
    else:
        char_key = character.lower()
        char_dict = characters[char_key]

        desc = "\n".join([f"- **{{{key}}}**" for key in char_dict.keys()])

        return desc
